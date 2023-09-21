import argparse
from utils.util import yaml_parser
from logger.logger import get_logger
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from model.build_model import get_model
from data.vocab import save_and_retrieve_vocab
from data.dataset import CustomCocoDataset, collate_function
from utils.image_utils import reshape_all_images
from loss.build_loss import get_loss


def get_args():
    parser = argparse.ArgumentParser(description="val")
    parser.add_argument(
        "--yaml", default="config/train.yaml", type=str, help="output model name"
    )
    parser.add_argument(
        "--model_path",
        default="train_log/2022-1018_1411/checkpoints/best.pth",
        type=str,
        help="output model name",
    )

    return parser.parse_args()


def evaluate(model, data_loader, device, cfg):
    loss_function = get_loss(cfg)  # create loss function
    mean_loss = torch.zeros(1).to(device)

    with torch.no_grad():
        model.eval()
        # data_loader = tqdm(data_loader, file=sys.stdout)

        for step, (imgs, caps, lens) in enumerate(data_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            tgts = pack_padded_sequence(caps, lens, batch_first=True)[0]
            pred = model(imgs, caps, lens)

            loss = loss_function(pred, tgts)

            # running average of the loss over multiple mini-batches
            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        return mean_loss.item()


if __name__ == "__main__":
    args = get_args()
    cfg = yaml_parser(args.yaml)

    logger = get_logger(name="train")  # log message printing

    vocab = save_and_retrieve_vocab(cfg, logger)
    reshape_all_images(cfg, logger)

    if cfg.GLOBAL.DEVICE == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif cfg.GLOBAL.DEVICE:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GLOBAL.DEVICE
        assert torch.cuda.is_available()
    cuda = cfg.GLOBAL.DEVICE != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    net = get_model(cfg, vocabulary=vocab)
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint["state_dict_backbone"])
    net.to(device)

    val_dataset = CustomCocoDataset(cfg=cfg, mode="val", vocabulary=vocab)
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCHSIZE_PER_CARD,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        drop_last=cfg.TRAIN.DROP_LAST,
        collate_fn=collate_function,
    )

    # evaluate validation set
    cross_entropy_loss = evaluate(net, val_loader, device)
    print("ce_loss", cross_entropy_loss)
