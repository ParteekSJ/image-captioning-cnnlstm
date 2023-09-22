import random
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


def read_split_data(cfg, mode):
    # traverse folders, one per category
    classes = [cla for cla in os.listdir(cfg.GLOBAL.TRAIN_DIR)]
    # Sort to ensure consistent order
    classes.sort()
    # Generate category names and corresponding numerical indexes
    class_indices = dict((k, v) for v, k in enumerate(classes))

    images_path = []
    images_label = []
    every_class_num = []

    if mode == "train":
        # iterates over each class (folder) in the classes list
        for cla in classes:
            # For each class, it constructs a list of image file paths (images) by listing the files in the corresponding subdirectory of cfg.GLOBAL.TRAIN_DIR
            train_cla_path = os.path.join(cfg.GLOBAL.TRAIN_DIR, cla)
            images = [
                os.path.join(cfg.GLOBAL.TRAIN_DIR, cla, i)
                for i in os.listdir(train_cla_path)
            ]
            # retrieves the numerical index (image_class) corresponding to the current class name using the class_indices dictionary
            image_class = class_indices[cla]
            # Record the number of pictures in each category
            every_class_num.append(len(images))
            # appends the image file paths (images_path) and their corresponding class labels (images_label) to their respective lists
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)
    elif mode == "val":
        for cla in classes:
            val_cal_path = os.path.join(cfg.GLOBAL.VAL_DIR, cla)
            # same steps but different directory
            images = [
                os.path.join(cfg.GLOBAL.VAL_DIR, cla, i)
                for i in os.listdir(val_cal_path)
            ]
            image_class = class_indices[cla]
            every_class_num.append(len(images))
            for img_path in images:
                images_path.append(img_path)
                images_label.append(image_class)
    """
    every_class_num - list containing the number of images in each class (useful for tracking class distribution).
    images_path - list of file paths to the images.
    images_label - list of numerical class labels corresponding to the images in images_path.
    """
    return every_class_num, images_path, images_label


def plot_image(num_classes, every_class_num, experiment_dir, mode):
    # plots the data in the every_class_num list as vertical bars
    plt.bar(range(len(num_classes)), every_class_num, align="center")
    plt.xticks(range(len(num_classes)), num_classes)
    for i, v in enumerate(every_class_num):
        plt.text(x=i, y=v + 5, s=str(v), ha="center")

    if mode == "train":
        plt.xlabel("train image class")
    elif mode == "val":
        plt.xlabel("val image class")

    plt.ylabel("number of images")
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
