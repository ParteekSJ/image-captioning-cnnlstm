import argparse
import datetime
import math
import random
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from shutil import copyfile
import json
from pathlib import Path
from collections import OrderedDict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def yaml_parser(yaml_path):
    with open(yaml_path, "r") as file:
        opt = argparse.Namespace(**yaml.load(file.read(), Loader=yaml.FullLoader))
    opt.GLOBAL = argparse.Namespace(**opt.GLOBAL)
    opt.TRAIN = argparse.Namespace(**opt.TRAIN)
    opt.DATASET = argparse.Namespace(**opt.DATASET)
    opt.MODEL = argparse.Namespace(**opt.MODEL)
    opt.CRITERION = argparse.Namespace(**opt.CRITERION)
    opt.OPTIMIZER = argparse.Namespace(**opt.OPTIMIZER)

    return opt


def init_setting(cfg):
    timestr = str(datetime.datetime.now().strftime("%Y-%m%d_%H%M"))
    experiment_dir = Path(cfg.GLOBAL.SAVE_RESULT_DIR)
    experiment_dir.mkdir(exist_ok=True)  # directory for saving experimental results
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)  # root directory of each experiment
    checkpoints_dir = experiment_dir.joinpath("checkpoints/")
    checkpoints_dir.mkdir(exist_ok=True)  # directory for saving the model
    tensorboard_dir = experiment_dir.joinpath("tensorboard/")
    tensorboard_dir.mkdir(exist_ok=True)  # directory for saving the logs
    setting_dir = experiment_dir.joinpath("setting/")
    setting_dir.mkdir(exist_ok=True)  # directory for saving the settings
    log_dir = experiment_dir.joinpath("log/")
    log_dir.mkdir(exist_ok=True)  # directory for saving the settings

    # copy various project files into the "setting" directory
    copyfile("data/dataset.py", str(setting_dir) + "/dataset.py")

    if cfg.GLOBAL.RESUME:
        copyfile(
            "config/retrain.yaml", str(setting_dir) + "/retrain.yaml"
        )  # retraining
    else:
        copyfile(
            "config/train.yaml", str(setting_dir) + "/train.yaml"
        )  # fresh training

    copyfile("loss/build_loss.py", str(setting_dir) + "/build_loss.py")
    copyfile("model/build_model.py", str(setting_dir) + "/build_model.py")
    copyfile("train.py", str(setting_dir) + "/train.py")
    copyfile("val.py", str(setting_dir) + "/val.py")

    # returns several directory paths
    return experiment_dir, checkpoints_dir, tensorboard_dir, log_dir


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Format numbers with commas for better readability
    total_num_str = "{:,}".format(total_num)
    trainable_num_str = "{:,}".format(trainable_num)

    info = f"Total: {total_num_str} params, Trainable: {trainable_num_str} params"
    return info


def count_optimizer_parameters(optimizer):
    total_params = 0

    for param_group in optimizer.param_groups:
        params = param_group["params"]
        total_params += sum(p.numel() for p in params)

    return total_params


def read_split_data(cfg, mode):
    # 遍历文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(cfg.GLOBAL.TRAIN_DIR)]
    # 排序，保证顺序一致
    classes.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))

    images_path = []
    images_label = []
    every_class_num = []

    if mode == "train":
        for cla in classes:
            train_cla_path = os.path.join(cfg.GLOBAL.TRAIN_DIR, cla)
            images = [
                os.path.join(cfg.GLOBAL.TRAIN_DIR, cla, i)
                for i in os.listdir(train_cla_path)
            ]
            image_class = class_indices[cla]
            every_class_num.append(len(images))  # 记录每个类别下的图片个数
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)
    elif mode == "val":
        for cla in classes:
            val_cal_path = os.path.join(cfg.GLOBAL.VAL_DIR, cla)
            # 遍历获取supported支持的所有文件路径
            images = [
                os.path.join(cfg.GLOBAL.VAL_DIR, cla, i)
                for i in os.listdir(val_cal_path)
            ]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(images))
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)

    return every_class_num, images_path, images_label


def plot_image(num_classes, every_class_num, experiment_dir, mode):
    plt.bar(range(len(num_classes)), every_class_num, align="center")
    # 将横坐标0,1,2,3,4替换为相应的类别名称
    plt.xticks(range(len(num_classes)), num_classes)
    # 在柱状图上添加数值标签
    for i, v in enumerate(every_class_num):
        plt.text(x=i, y=v + 5, s=str(v), ha="center")
    # 设置x坐标
    if mode == "train":
        plt.xlabel("train image class")
    elif mode == "val":
        plt.xlabel("val image class")
    # 设置y坐标
    plt.ylabel("number of images")
    # 设置柱状图的标题
    plt.title("class distribution")
    if mode == "train":
        plt.savefig(os.path.join(experiment_dir, "train_dataset.png"))
        plt.close()
    elif mode == "val":
        plt.savefig(os.path.join(experiment_dir, "val_dataset.png"))
        plt.close()


def view_dataset(experiment_dir, cfg):
    img_list = glob.glob(cfg.GLOBAL.TRAIN_DIR + "/*/*.jpg")
    train_num_classes = os.listdir(cfg.GLOBAL.TRAIN_DIR)
    train_num_classes.sort()  # 排序，为了每次排序的结果一致
    val_num_classes = os.listdir(cfg.GLOBAL.VAL_DIR)
    val_num_classes.sort()
    random.shuffle(img_list)
    img_list = img_list[:9]
    transform_list = [
        transforms.Resize((cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE)),
        transforms.RandomCrop((cfg.TRAIN.IMG_SIZE, cfg.TRAIN.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=cfg.TRAIN.TRANSFORMS_BRIGHTNESS,
            contrast=cfg.TRAIN.TRANSFORMS_CONTRAST,
            saturation=cfg.TRAIN.TRANSFORMS_SATURATION,
            hue=cfg.TRAIN.TRANSFORMS_HUE,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform_list)
    test_img_list = []
    for pic_path in img_list:
        test_img = Image.open(pic_path)
        test_img_list.append(test_img)
    nrows = 3
    ncols = 3
    figsize = (8, 8)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            img = transform(test_img_list[i + j])  # 选取3x3=9张图片进行数据增强可视化
            img = img.numpy().transpose((1, 2, 0))
            img = np.clip(img, 0, 1)
            figs[i][j].imshow(img)
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(experiment_dir, "dataset_aug.png"))
    plt.close()

    train_every_class_num, _, _ = read_split_data(cfg, "train")  # 读取数据集
    val_every_class_num, _, _ = read_split_data(cfg, "val")
    plot_image(train_num_classes, train_every_class_num, experiment_dir, "train")  # 画图
    plot_image(val_num_classes, val_every_class_num, experiment_dir, "val")


def build_scheduler(optimizer, cfg):
    epochs = cfg.GLOBAL.EPOCH_NUM

    if cfg.OPTIMIZER.LR_NAME == "none":
        # Return a scheduler with a constant learning rate
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda x: 1.0,
        )
    elif cfg.OPTIMIZER.LR_NAME == "linear_lr":
        lf = (
            lambda x: (1 - x / (epochs - 1)) * (1.0 - cfg.OPTIMIZER.LR_DECAY)
            + cfg.OPTIMIZER.LR_DECAY
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lf,
        )
    elif cfg.OPTIMIZER.LR_NAME == "cosine_lr":
        lf = (
            lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2)
            * (1.0 - cfg.OPTIMIZER.LR_DECAY)
            + cfg.OPTIMIZER.LR_DECAY
        )
        scheduler = lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lf,
        )

    return scheduler


# def build_optimizer(model, cfg, logger):
#     if cfg.OPTIMIZER.NAME == "Adam":
#         optimizer = Adam(params=model.params, lr=cfg.OPTIMIZER.LEARNING_RATE)
#         return optimizer
#     return None


def build_optimizer(model, cfg, logger):
    """
    g0 - store weights of batch normalization layers which do not have any weight decays
    g1 - store weights (not from batch normalization layers) which will have weight decays
    g2 - store biases
    """

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    total_params = 0  # Initialize total parameter count

    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(
            v.weight, nn.Parameter
        ):  # weight (with decay)
            g1.append(v.weight)

    # Accumulate the total number of parameters
    total_params = sum(p.numel() for group in [g0, g1, g2] for p in group)

    if cfg.OPTIMIZER.NAME == "Adam":
        optimizer = Adam(
            g0,
            lr=cfg.OPTIMIZER.LEARNING_RATE,
            betas=[cfg.OPTIMIZER.BETA1, cfg.OPTIMIZER.BETA2],
        )
        optimizer.add_param_group(
            {"params": g1, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}
        )  # add g1 with weight_decay
        optimizer.add_param_group({"params": g2})  # add g2 (biases)

    elif cfg.OPTIMIZER.NAME == "SGD":
        optimizer = SGD(
            g0,
            lr=cfg.OPTIMIZER.LEARNING_RATE,
            momentum=cfg.OPTIMIZER.MOMENTUM,
            nesterov=cfg.OPTIMIZER.NESTEROV,
        )
        optimizer.add_param_group(
            {"params": g1, "weight_decay": cfg.OPTIMIZER.WEIGHT_DECAY}
        )  # add g1 with weight_decay
        optimizer.add_param_group({"params": g2})  # add g2 (biases)

    # logger.info(
    #     f"{'optimizer:'} {type(optimizer).__name__} with parameter groups "
    #     f"g0: {len(g0)} weight, g1: {len(g1)} weight (no decay), g2: {len(g2)} bias"
    # )
    del g0, g1, g2

    return optimizer


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)