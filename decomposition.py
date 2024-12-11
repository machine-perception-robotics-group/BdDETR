import os
import csv
import sys
import argparse
import torch
import torch.nn as nn
import time
import pathlib

from tqdm import tqdm
from timm.models import create_model
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import utils
from models import detr as detr
from bdnn_module.utils import extract_weight_ext as ew
from bdnn_module.utils import calc_comput_complex as cFT


def args():
    parser = argparse.ArgumentParser('DeiT evaluation script', add_help=False)
    parser.add_argument('--img_path', default='~/data/imagenet/val_IF_WID', type=str, metavar='MODEL',
                        help='imagenet dataset path')
    parser.add_argument('--model', default='detr', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--weights', default='./weights/deit_base/deit_base_patch16_224-b5f2ef4d.pth',
                        help='weights path')
    parser.add_argument('--device', default='cpu',
                        help='device to use for testing')
    parser.add_argument('--qb', default='8', type=int)
    
    return parser.parse_args()


def main(args):
    save_dir = os.path.join(os.path.dirname(args.weights), 'binary7565')
    device = torch.device(args.device)

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'])
    range_basis = [args.qb]
    mode = 'exh'
    
    ew.save_param(model, range_basis, mode, save_dir, w_param = True)


if __name__ == '__main__':
    main(args())
