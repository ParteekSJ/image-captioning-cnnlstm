import argparse
import datetime
import random
import os
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from shutil import copyfile
from pathlib import Path
from constants import BASE_DIR


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


def yaml_parser(yaml_path):
    # Load the YAML file as plain text
    with open(yaml_path, "r") as file:
        yaml_text = file.read()

    # Perform variable substitution for ${BASE_DIR}
    yaml_text = yaml_text.replace("${BASE_DIR}", BASE_DIR)

    # Parse the modified YAML text
    config = yaml.safe_load(yaml_text)

    # Create the argparse.Namespace object
    opt = argparse.Namespace(**config)
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
    copyfile(
        os.path.join(BASE_DIR, "data/dataset.py"), str(setting_dir) + "/dataset.py"
    )

    if cfg.GLOBAL.RESUME:
        copyfile(
            os.path.join(BASE_DIR, "config/retrain.yaml"),
            str(setting_dir) + "/retrain.yaml",
        )  # retraining
    else:
        copyfile(
            os.path.join(BASE_DIR, "config/train.yaml"),
            str(setting_dir) + "/train.yaml",
        )  # fresh training

    copyfile(
        os.path.join(BASE_DIR, "loss/build_loss.py"),
        str(setting_dir) + "/build_loss.py",
    )
    copyfile(
        os.path.join(BASE_DIR, "model/build_model.py"),
        str(setting_dir) + "/build_model.py",
    )
    copyfile(os.path.join(BASE_DIR, "train.py"), str(setting_dir) + "/train.py")
    copyfile(os.path.join(BASE_DIR, "val.py"), str(setting_dir) + "/val.py")

    # returns several directory paths
    return experiment_dir, checkpoints_dir, tensorboard_dir, log_dir
