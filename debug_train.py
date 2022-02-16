"""
debug to test training
"""

import sys
sys.path.append("/home/rchoudhu/research/voxelpose-pytorch/lib")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import pprint
import logging
import json
import time
#%matplotlib agg
import matplotlib.pyplot as plt

from tqdm import tqdm

#import _init_paths
from core.config import config
from core.config import update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_model_state
from utils.utils import load_backbone_panoptic
from utils.vis import save_debug_3d_images
import dataset
import models

def get_optimizer(model):
    lr = config.TRAIN.LR
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
    for params in model.module.root_net.parameters():
        params.requires_grad = True
    for params in model.module.pose_net.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer

# Prep the config
cfg = "configs/panoptic/resnet50/prn64_cpn80x80x20_960x512_cam5.yaml"
update_config(cfg)
logger, final_output_dir, tb_log_dir = create_logger(config, cfg, 'train')

gpus = [int(i) for i in config.GPUS.split(',')]
print('=> Loading data ..')
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Make dataloaders
train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
    shuffle=config.TRAIN.SHUFFLE,
    num_workers=config.WORKERS,
    pin_memory=True)

test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
    config, config.DATASET.TEST_SUBSET, False,
    transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=config.TEST.BATCH_SIZE * len(gpus),
    shuffle=False,
    num_workers=config.WORKERS,
    pin_memory=True)

cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

print('=> Constructing models ..')
model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
    config, is_train=True)

logger.info("Setting data parallel with gpus: " + str(gpus))
gpus=[0]
start_time = time.time()
with torch.no_grad():
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
logger.info("Took %.3f to set up data parallel")

logger.info("Getting optimizer ... ")
model, optimizer = get_optimizer(model)

start_epoch = config.TRAIN.BEGIN_EPOCH
end_epoch = config.TRAIN.END_EPOCH

best_precision = 0
if config.NETWORK.PRETRAINED_BACKBONE:
    logger.info("loading panoptic backbone")
    panoptic_load_start_time = time.time()
    model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
    logger.info("Took %.3f to load panoptic backbone." % (time.time() - panoptic_load_start_time))
if config.TRAIN.RESUME:
    start_epoch, model, optimizer, best_precision = load_checkpoint(model, optimizer, final_output_dir)

writer_dict = {
    'writer': SummaryWriter(log_dir=tb_log_dir),
    'train_global_steps': 0,
    'valid_global_steps': 0,
}

print("Ready to train")