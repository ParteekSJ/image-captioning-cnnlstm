import argparse, os, sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
from logger.logger import get_logger

# from tqdm import tqdm
from logger.logger import setup_logging
from utils.image_utils import reshape_all_images
from utils.util import (
    yaml_parser,
    init_setting,
    setup_seed,
    get_parameter_number,
    build_optimizer,
    build_scheduler,
)
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence
from data.dataset import CustomCocoDataset
from data.vocab import save_and_retrieve_vocab
from data.dataset import collate_function
from model.build_model import get_model, parallel_model, de_parallel
from loss.build_loss import get_loss
from copy import deepcopy
from val import evaluate


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--yaml",
        default="config/train.yaml",
        type=str,
        help="configuration file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="DDP parameter",
    )

    return parser.parse_args()


def get_envs():
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    rank = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def train(cfg):
    # root directories of the training experiment, checkpoints, tensorboard_logs, and logs(text)
    experiment_dir, checkpoints_dir, tensorboard_dir, log_dir = init_setting(cfg)
    setup_logging(save_dir=log_dir)
    setup_seed(2022)

    logger = get_logger(name="train")  # log message printing
    vocab = save_and_retrieve_vocab(cfg, logger)
    reshape_all_images(cfg, logger)

    # setup enviroment
    if cfg.GLOBAL.DEVICE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif cfg.GLOBAL.DEVICE:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()
    cuda = cfg.GLOBAL.DEVICE != "cpu" and torch.cuda.is_available()  # bool
    device = torch.device("cuda:0" if cuda else "cpu")
    rank, local_rank, world_size = get_envs()

    if local_rank != -1:  # DDP distributed mode
        logger.info("In DDP mode.")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            init_method="env://",
            rank=local_rank,
            world_size=world_size,
        )

    tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(
        f'Start Tensorboard with "tensorboard --logdir={tensorboard_dir}", view at http://localhost:6006/'
    )

    train_dataset = CustomCocoDataset(cfg=cfg, mode="train", vocabulary=vocab)
    logger.info(f"Training dataset created with {len(train_dataset)} samples.")
    train_sampler = (
        None
        if rank == -1
        else distributed.DistributedSampler(train_dataset, shuffle=True)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCHSIZE_PER_CARD // world_size,
        shuffle=True and train_sampler is None,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        sampler=train_sampler,
        drop_last=cfg.TRAIN.DROP_LAST,
        collate_fn=collate_function,
    )
    logger.info(f"Training dataloader created with {len(train_loader)} batches.")

    if rank in [-1, 0]:
        val_dataset = CustomCocoDataset(cfg=cfg, mode="val", vocabulary=vocab)
        logger.info(f"Validation dataset created with {len(val_dataset)} samples.")
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.TRAIN.BATCHSIZE_PER_CARD,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            drop_last=cfg.TRAIN.DROP_LAST,
            collate_fn=collate_function,
        )
        logger.info(f"Validation dataloader created with {len(val_loader)} batches.")

    net = get_model(cfg, vocab)  # create model
    logger.info(get_parameter_number(net))  # calculate how many parameters
    net = net.to(device)

    optimizer = build_optimizer(net, cfg, logger)  # create optimizer
    scheduler = build_scheduler(optimizer, cfg)  # create lr scheduler
    loss_function = get_loss(cfg)  # create loss function
    start_epoch = 0

    # IF RESUMING TRAINING
    if cfg.GLOBAL.RESUME:
        checkpoint = torch.load(cfg.GLOBAL.RESUME_PATH)
        logger.info("loading checkpoint from {}".format(cfg.GLOBAL.RESUME_PATH))
        start_epoch = checkpoint["epoch"]
        state_dict = checkpoint["state_dict_backbone"]
        net.load_state_dict(state_dict, strict=False)
        state_optimizer = checkpoint["state_optimizer"]
        optimizer.load_state_dict(state_optimizer)
        state_lr_scheduler = checkpoint["state_lr_scheduler"]
        scheduler.load_state_dict(state_lr_scheduler)

    # create parallel model
    net = parallel_model(net, device, rank, local_rank)

    scaler = GradScaler(enabled=cfg.GLOBAL.USE_AMP)  # Mixed Precision Training

    pre_ce_loss = 100.0  # starting loss
    early_stop_patience = 0

    # FIX (start_epoch + cfg.GLOBAL.EPOCH_NUM): to ensure we train for as many epochs specified in the yaml file
    for epoch in range(start_epoch, start_epoch + cfg.GLOBAL.EPOCH_NUM):
        net.train()
        if rank != -1:  # if in DDP mode
            train_loader.sampler.set_epoch(epoch)
        mean_loss = torch.zeros(1).to(device)

        data_loader = train_loader
        # data_loader = tqdm(train_loader, file=sys.stdout)
        # file=sys.stdout - progress bar should be displayed in the standard output (console).

        for step, (imgs, caps, lens) in enumerate(data_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.GLOBAL.USE_AMP):
                # Set mini-batch dataset
                imgs = imgs.to(device)
                caps = caps.to(device)
                tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]

                pred = net(imgs, caps, lens)
                loss = loss_function(pred, tgts)

                # distributed synchronization and reduction of loss
                if rank != -1:
                    dist.all_reduce(loss)  # Synchronize and sum losses across GPUs
                    loss /= world_size  # Compute the mean loss
                    # loss *= world_size

                # running average of the loss over multiple mini-batches
                mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

                if step % 5 == 0 and rank in [-1, 0]:
                    logger.info(
                        f"Epoch [{epoch + 1}/{cfg.GLOBAL.EPOCH_NUM}], Step[{step + 1}/{len(data_loader)}], Loss: {mean_loss.item():.4f}"
                    )

                # data_loader.desc = "[epoch {}] mean train loss {}".format(
                #     epoch, round(mean_loss.item(), 3)
                # )

                if not torch.isfinite(loss):  # if NaN
                    logger.warning("WARNING: non-finite loss, ending training ", loss)
                    sys.exit(1)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        # # Saving training loss values & learning rates to tensorboard
        tb_writer.add_scalar(
            tag="training_loss",
            scalar_value=mean_loss.item(),
            global_step=epoch,
        )

        tb_writer.add_scalar(
            tag="learning_rate",
            scalar_value=optimizer.param_groups[0]["lr"],
            global_step=epoch,
        )

        tb_writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={"training_loss": mean_loss.item()},
            global_step=epoch,
        )

        scheduler.step()

        # VALIDATION (every VAL_EPOCH_STEP(s)) only ranks (-1 or 0)
        if epoch % cfg.GLOBAL.VAL_EPOCH_STEP == 0 and rank in [-1, 0]:
            ce_loss = evaluate(
                model=net, data_loader=val_loader, device=device, cfg=cfg
            )

            logger.info(
                f"Validation Epoch: [{epoch + 1}/{cfg.GLOBAL.EPOCH_NUM}] Loss: {round(ce_loss, 3)}"
            )

            # Saving validation loss values to tensorboard
            tb_writer.add_scalar(
                tag="validation_loss",
                scalar_value=ce_loss,
                global_step=epoch,
            )

            tb_writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict={"validation_loss": ce_loss},
                global_step=epoch,
            )

        if epoch % cfg.GLOBAL.SAVE_EPOCH_STEP == 0 and rank in [-1, 0]:
            checkpoint = {
                "epoch": epoch,
                "state_dict_backbone": deepcopy(de_parallel(net)).state_dict(),
                "state_optimizer": optimizer.state_dict(),
                "state_lr_scheduler": scheduler.state_dict(),
            }  # save state dictionary
            torch.save(checkpoint, checkpoints_dir / f"model-{epoch}.pth")

            # if current NLL loss is lesser than previous NLL loss, save it
            if ce_loss < pre_ce_loss:
                torch.save(checkpoint, checkpoints_dir / "best.pth".format(epoch))
                logger.info("Model Saved.")
                pre_ce_loss = ce_loss
                early_stop_patience = 0
            else:
                early_stop_patience += 1
                if early_stop_patience > cfg.GLOBAL.EARLY_STOP_PATIENCE:
                    logger.info(
                        # "acc exceeds times without improvement, stopped training early"
                        "no decrease in cross entropy loss, stopped training early"
                    )

                    # destroy process
                    if world_size > 1 and rank == 0:
                        dist.destroy_process_group()

                    sys.exit(1)

    # destroy process
    if world_size > 1 and rank == 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = get_args()  # getting the input arguments
    cfg = yaml_parser(args.yaml)  # parse the configuration file
    train(cfg)
