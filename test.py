import argparse
from math import fabs
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

import os
import csv
import sys
import pathlib
import numpy as np
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from binary_engine import evaluate
from models_bd import build_model
num_threads = 16


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # bdnn
    parser.add_argument('--weights', default='./weights/detr/detr-r50-e632da11.pth',
                        help='weights path')
    parser.add_argument('--binary_weights', default='binary7565',
                        help='binary weights path')
    parser.add_argument('--qb', default=8, type=int)
    
    
    return parser


def text2list(file_path, name):
    with open(file_path, 'r') as f:
        return [row for row in csv.reader(f) if name in row]

def calc_param_size(weight, dtype):
    weight_size = weight.size
    if weight_size < 1:
        raise ValueError("weight shape must be 4 or 2 dimension.")
    return (weight_size * dtype) / (8 * 1024 ** 2)


def calc_params(net, layer_lef, size_param):
    layer_ref = layer_lef
    list_size_bias_param = []  # Bias
    for name, param in net.state_dict().items():
        weight_dim = net.state_dict()[name].dim()
        weight_np = net.state_dict()[name].cpu().clone().detach().numpy()
        weight_np = weight_np.astype(np.float64)
        if (weight_np.size > 0) and (weight_dim == 1):
            list_size_bias_param.append(calc_param_size(weight_np, 32) / 1024.0)

    size_decomp_param = sum([x[3] for x in layer_ref]) # Decomposed parameters
    size_bias_param   = sum([x for x in list_size_bias_param])

    size_decomp_paramWbias = size_decomp_param + size_bias_param
    size_paramWbias = size_param + size_bias_param
    compress_ratio = 100 * (size_decomp_param / size_param)
    compress_ratioWbias = 100 * (size_paramWbias / size_decomp_paramWbias)

    print(f"TotalParams [MB]: {size_paramWbias: 4f}  DecomposedParams [MB]: {size_decomp_paramWbias: 4f}  Bias [MB]: {size_bias_param: 4f}")
    print(f"CompressRatio: {compress_ratio: 4f}  CompressRatioWbias: {compress_ratioWbias: 4f}")

def load_param(net, Q_bits_list, layer_basis, param_dir, d_mode='exh'):
    basis_list = [os.path.join(param_dir, f'B{k}/') for k in layer_basis]
    param_info_path = [os.path.join(param_dir, 'B' + str(k) + '/param_info_' + d_mode + '.csv') for k in layer_basis]
    total_param = float(text2list(param_info_path[0], 'ALL')[0][2])

    print('+-----Load parameters-----+')
    net.transformer.encoder.layers[0].self_attn.M_ui_np = np.load(os.path.join(basis_list[0], 'transformer.encoder.layers.0.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[0].self_attn.c_np    = np.load(os.path.join(basis_list[0], 'transformer.encoder.layers.0.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.encoder.layers[0].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[1], 'transformer.encoder.layers.0.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[0].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[1], 'transformer.encoder.layers.0.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.encoder.layers[0].linear1.M_ui_np = np.load(os.path.join(basis_list[2], 'transformer.encoder.layers.0.linear1.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[0].linear1.c_np    = np.load(os.path.join(basis_list[2], 'transformer.encoder.layers.0.linear1.c_exh.pth.npy'))
    net.transformer.encoder.layers[0].linear2.M_ui_np = np.load(os.path.join(basis_list[3], 'transformer.encoder.layers.0.linear2.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[0].linear2.c_np    = np.load(os.path.join(basis_list[3], 'transformer.encoder.layers.0.linear2.c_exh.pth.npy'))
    net.transformer.encoder.layers[1].self_attn.M_ui_np = np.load(os.path.join(basis_list[4], 'transformer.encoder.layers.1.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[1].self_attn.c_np    = np.load(os.path.join(basis_list[4], 'transformer.encoder.layers.1.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.encoder.layers[1].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[5], 'transformer.encoder.layers.1.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[1].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[5], 'transformer.encoder.layers.1.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.encoder.layers[1].linear1.M_ui_np = np.load(os.path.join(basis_list[6], 'transformer.encoder.layers.1.linear1.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[1].linear1.c_np    = np.load(os.path.join(basis_list[6], 'transformer.encoder.layers.1.linear1.c_exh.pth.npy'))
    net.transformer.encoder.layers[1].linear2.M_ui_np = np.load(os.path.join(basis_list[7], 'transformer.encoder.layers.1.linear2.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[1].linear2.c_np    = np.load(os.path.join(basis_list[7], 'transformer.encoder.layers.1.linear2.c_exh.pth.npy'))
    net.transformer.encoder.layers[2].self_attn.M_ui_np = np.load(os.path.join(basis_list[8], 'transformer.encoder.layers.2.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[2].self_attn.c_np    = np.load(os.path.join(basis_list[8], 'transformer.encoder.layers.2.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.encoder.layers[2].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[9], 'transformer.encoder.layers.2.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[2].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[9], 'transformer.encoder.layers.2.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.encoder.layers[2].linear1.M_ui_np = np.load(os.path.join(basis_list[10], 'transformer.encoder.layers.2.linear1.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[2].linear1.c_np    = np.load(os.path.join(basis_list[10], 'transformer.encoder.layers.2.linear1.c_exh.pth.npy'))
    net.transformer.encoder.layers[2].linear2.M_ui_np = np.load(os.path.join(basis_list[11], 'transformer.encoder.layers.2.linear2.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[2].linear2.c_np    = np.load(os.path.join(basis_list[11], 'transformer.encoder.layers.2.linear2.c_exh.pth.npy'))
    net.transformer.encoder.layers[3].self_attn.M_ui_np = np.load(os.path.join(basis_list[12], 'transformer.encoder.layers.3.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[3].self_attn.c_np    = np.load(os.path.join(basis_list[12], 'transformer.encoder.layers.3.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.encoder.layers[3].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[13], 'transformer.encoder.layers.3.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[3].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[13], 'transformer.encoder.layers.3.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.encoder.layers[3].linear1.M_ui_np = np.load(os.path.join(basis_list[14], 'transformer.encoder.layers.3.linear1.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[3].linear1.c_np    = np.load(os.path.join(basis_list[14], 'transformer.encoder.layers.3.linear1.c_exh.pth.npy'))
    net.transformer.encoder.layers[3].linear2.M_ui_np = np.load(os.path.join(basis_list[15], 'transformer.encoder.layers.3.linear2.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[3].linear2.c_np    = np.load(os.path.join(basis_list[15], 'transformer.encoder.layers.3.linear2.c_exh.pth.npy'))
    net.transformer.encoder.layers[4].self_attn.M_ui_np = np.load(os.path.join(basis_list[16], 'transformer.encoder.layers.4.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[4].self_attn.c_np    = np.load(os.path.join(basis_list[16], 'transformer.encoder.layers.4.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.encoder.layers[4].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[17], 'transformer.encoder.layers.4.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[4].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[17], 'transformer.encoder.layers.4.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.encoder.layers[4].linear1.M_ui_np = np.load(os.path.join(basis_list[18], 'transformer.encoder.layers.4.linear1.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[4].linear1.c_np    = np.load(os.path.join(basis_list[18], 'transformer.encoder.layers.4.linear1.c_exh.pth.npy'))
    net.transformer.encoder.layers[4].linear2.M_ui_np = np.load(os.path.join(basis_list[19], 'transformer.encoder.layers.4.linear2.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[4].linear2.c_np    = np.load(os.path.join(basis_list[19], 'transformer.encoder.layers.4.linear2.c_exh.pth.npy'))
    net.transformer.encoder.layers[5].self_attn.M_ui_np = np.load(os.path.join(basis_list[20], 'transformer.encoder.layers.5.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[5].self_attn.c_np    = np.load(os.path.join(basis_list[20], 'transformer.encoder.layers.5.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.encoder.layers[5].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[21], 'transformer.encoder.layers.5.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[5].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[21], 'transformer.encoder.layers.5.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.encoder.layers[5].linear1.M_ui_np = np.load(os.path.join(basis_list[22], 'transformer.encoder.layers.5.linear1.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[5].linear1.c_np    = np.load(os.path.join(basis_list[22], 'transformer.encoder.layers.5.linear1.c_exh.pth.npy'))
    net.transformer.encoder.layers[5].linear2.M_ui_np = np.load(os.path.join(basis_list[23], 'transformer.encoder.layers.5.linear2.M_ui_exh.pth.npy'))
    net.transformer.encoder.layers[5].linear2.c_np    = np.load(os.path.join(basis_list[23], 'transformer.encoder.layers.5.linear2.c_exh.pth.npy'))
    
    net.transformer.decoder.layers[0].self_attn.M_ui_np = np.load(os.path.join(basis_list[24], 'transformer.decoder.layers.0.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[0].self_attn.c_np    = np.load(os.path.join(basis_list[24], 'transformer.decoder.layers.0.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[0].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[25], 'transformer.decoder.layers.0.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[0].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[25], 'transformer.decoder.layers.0.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[0].multihead_attn.M_ui_np = np.load(os.path.join(basis_list[26], 'transformer.decoder.layers.0.multihead_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[0].multihead_attn.c_np    = np.load(os.path.join(basis_list[26], 'transformer.decoder.layers.0.multihead_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[0].multihead_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[27], 'transformer.decoder.layers.0.multihead_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[0].multihead_attn.out_proj.c_np    = np.load(os.path.join(basis_list[27], 'transformer.decoder.layers.0.multihead_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[0].linear1.M_ui_np = np.load(os.path.join(basis_list[28], 'transformer.decoder.layers.0.linear1.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[0].linear1.c_np    = np.load(os.path.join(basis_list[28], 'transformer.decoder.layers.0.linear1.c_exh.pth.npy'))
    net.transformer.decoder.layers[0].linear2.M_ui_np = np.load(os.path.join(basis_list[29], 'transformer.decoder.layers.0.linear2.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[0].linear2.c_np    = np.load(os.path.join(basis_list[29], 'transformer.decoder.layers.0.linear2.c_exh.pth.npy'))
    net.transformer.decoder.layers[1].self_attn.M_ui_np = np.load(os.path.join(basis_list[30], 'transformer.decoder.layers.1.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[1].self_attn.c_np    = np.load(os.path.join(basis_list[30], 'transformer.decoder.layers.1.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[1].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[31], 'transformer.decoder.layers.1.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[1].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[31], 'transformer.decoder.layers.1.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[1].multihead_attn.M_ui_np = np.load(os.path.join(basis_list[32], 'transformer.decoder.layers.1.multihead_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[1].multihead_attn.c_np    = np.load(os.path.join(basis_list[32], 'transformer.decoder.layers.1.multihead_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[1].multihead_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[33], 'transformer.decoder.layers.1.multihead_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[1].multihead_attn.out_proj.c_np    = np.load(os.path.join(basis_list[33], 'transformer.decoder.layers.1.multihead_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[1].linear1.M_ui_np = np.load(os.path.join(basis_list[34], 'transformer.decoder.layers.1.linear1.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[1].linear1.c_np    = np.load(os.path.join(basis_list[34], 'transformer.decoder.layers.1.linear1.c_exh.pth.npy'))
    net.transformer.decoder.layers[1].linear2.M_ui_np = np.load(os.path.join(basis_list[35], 'transformer.decoder.layers.1.linear2.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[1].linear2.c_np    = np.load(os.path.join(basis_list[35], 'transformer.decoder.layers.1.linear2.c_exh.pth.npy'))
    net.transformer.decoder.layers[2].self_attn.M_ui_np = np.load(os.path.join(basis_list[36], 'transformer.decoder.layers.2.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[2].self_attn.c_np    = np.load(os.path.join(basis_list[36], 'transformer.decoder.layers.2.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[2].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[37], 'transformer.decoder.layers.2.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[2].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[37], 'transformer.decoder.layers.2.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[2].multihead_attn.M_ui_np = np.load(os.path.join(basis_list[38], 'transformer.decoder.layers.2.multihead_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[2].multihead_attn.c_np    = np.load(os.path.join(basis_list[38], 'transformer.decoder.layers.2.multihead_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[2].multihead_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[39], 'transformer.decoder.layers.2.multihead_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[2].multihead_attn.out_proj.c_np    = np.load(os.path.join(basis_list[39], 'transformer.decoder.layers.2.multihead_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[2].linear1.M_ui_np = np.load(os.path.join(basis_list[40], 'transformer.decoder.layers.2.linear1.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[2].linear1.c_np    = np.load(os.path.join(basis_list[40], 'transformer.decoder.layers.2.linear1.c_exh.pth.npy'))
    net.transformer.decoder.layers[2].linear2.M_ui_np = np.load(os.path.join(basis_list[41], 'transformer.decoder.layers.2.linear2.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[2].linear2.c_np    = np.load(os.path.join(basis_list[41], 'transformer.decoder.layers.2.linear2.c_exh.pth.npy'))
    net.transformer.decoder.layers[3].self_attn.M_ui_np = np.load(os.path.join(basis_list[42], 'transformer.decoder.layers.3.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[3].self_attn.c_np    = np.load(os.path.join(basis_list[42], 'transformer.decoder.layers.3.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[3].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[43], 'transformer.decoder.layers.3.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[3].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[43], 'transformer.decoder.layers.3.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[3].multihead_attn.M_ui_np = np.load(os.path.join(basis_list[44], 'transformer.decoder.layers.3.multihead_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[3].multihead_attn.c_np    = np.load(os.path.join(basis_list[44], 'transformer.decoder.layers.3.multihead_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[3].multihead_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[45], 'transformer.decoder.layers.3.multihead_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[3].multihead_attn.out_proj.c_np    = np.load(os.path.join(basis_list[45], 'transformer.decoder.layers.3.multihead_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[3].linear1.M_ui_np = np.load(os.path.join(basis_list[46], 'transformer.decoder.layers.3.linear1.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[3].linear1.c_np    = np.load(os.path.join(basis_list[46], 'transformer.decoder.layers.3.linear1.c_exh.pth.npy'))
    net.transformer.decoder.layers[3].linear2.M_ui_np = np.load(os.path.join(basis_list[47], 'transformer.decoder.layers.3.linear2.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[3].linear2.c_np    = np.load(os.path.join(basis_list[47], 'transformer.decoder.layers.3.linear2.c_exh.pth.npy'))
    net.transformer.decoder.layers[4].self_attn.M_ui_np = np.load(os.path.join(basis_list[48], 'transformer.decoder.layers.4.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[4].self_attn.c_np    = np.load(os.path.join(basis_list[48], 'transformer.decoder.layers.4.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[4].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[49], 'transformer.decoder.layers.4.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[4].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[49], 'transformer.decoder.layers.4.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[4].multihead_attn.M_ui_np = np.load(os.path.join(basis_list[50], 'transformer.decoder.layers.4.multihead_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[4].multihead_attn.c_np    = np.load(os.path.join(basis_list[50], 'transformer.decoder.layers.4.multihead_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[4].multihead_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[51], 'transformer.decoder.layers.4.multihead_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[4].multihead_attn.out_proj.c_np    = np.load(os.path.join(basis_list[51], 'transformer.decoder.layers.4.multihead_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[4].linear1.M_ui_np = np.load(os.path.join(basis_list[52], 'transformer.decoder.layers.4.linear1.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[4].linear1.c_np    = np.load(os.path.join(basis_list[52], 'transformer.decoder.layers.4.linear1.c_exh.pth.npy'))
    net.transformer.decoder.layers[4].linear2.M_ui_np = np.load(os.path.join(basis_list[53], 'transformer.decoder.layers.4.linear2.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[4].linear2.c_np    = np.load(os.path.join(basis_list[53], 'transformer.decoder.layers.4.linear2.c_exh.pth.npy'))
    net.transformer.decoder.layers[5].self_attn.M_ui_np = np.load(os.path.join(basis_list[54], 'transformer.decoder.layers.5.self_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[5].self_attn.c_np    = np.load(os.path.join(basis_list[54], 'transformer.decoder.layers.5.self_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[5].self_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[55], 'transformer.decoder.layers.5.self_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[5].self_attn.out_proj.c_np    = np.load(os.path.join(basis_list[55], 'transformer.decoder.layers.5.self_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[5].multihead_attn.M_ui_np = np.load(os.path.join(basis_list[56], 'transformer.decoder.layers.5.multihead_attn.in_proj_weight.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[5].multihead_attn.c_np    = np.load(os.path.join(basis_list[56], 'transformer.decoder.layers.5.multihead_attn.in_proj_weight.c_exh.pth.npy'))
    net.transformer.decoder.layers[5].multihead_attn.out_proj.M_ui_np = np.load(os.path.join(basis_list[57], 'transformer.decoder.layers.5.multihead_attn.out_proj.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[5].multihead_attn.out_proj.c_np    = np.load(os.path.join(basis_list[57], 'transformer.decoder.layers.5.multihead_attn.out_proj.c_exh.pth.npy'))
    net.transformer.decoder.layers[5].linear1.M_ui_np = np.load(os.path.join(basis_list[58], 'transformer.decoder.layers.5.linear1.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[5].linear1.c_np    = np.load(os.path.join(basis_list[58], 'transformer.decoder.layers.5.linear1.c_exh.pth.npy'))
    net.transformer.decoder.layers[5].linear2.M_ui_np = np.load(os.path.join(basis_list[59], 'transformer.decoder.layers.5.linear2.M_ui_exh.pth.npy'))
    net.transformer.decoder.layers[5].linear2.c_np    = np.load(os.path.join(basis_list[59], 'transformer.decoder.layers.5.linear2.c_exh.pth.npy'))
    
    
    net.class_embed.M_ui_np = np.load(os.path.join(basis_list[60], 'class_embed.M_ui_exh.pth.npy'))
    net.class_embed.c_np    = np.load(os.path.join(basis_list[60], 'class_embed.c_exh.pth.npy'))

    net.bbox_embed.layers[0].M_ui_np = np.load(os.path.join(basis_list[61], 'bbox_embed.layers.0.M_ui_exh.pth.npy'))
    net.bbox_embed.layers[0].c_np    = np.load(os.path.join(basis_list[61], 'bbox_embed.layers.0.c_exh.pth.npy'))
    net.bbox_embed.layers[1].M_ui_np = np.load(os.path.join(basis_list[62], 'bbox_embed.layers.1.M_ui_exh.pth.npy'))
    net.bbox_embed.layers[1].c_np    = np.load(os.path.join(basis_list[62], 'bbox_embed.layers.1.c_exh.pth.npy'))
    net.bbox_embed.layers[2].M_ui_np = np.load(os.path.join(basis_list[63], 'bbox_embed.layers.2.M_ui_exh.pth.npy'))
    net.bbox_embed.layers[2].c_np    = np.load(os.path.join(basis_list[63], 'bbox_embed.layers.2.c_exh.pth.npy'))

    net.query_embed.M_ui_np = np.load(os.path.join(basis_list[64], 'query_embed.M_ui_exh.pth.npy'))
    net.query_embed.c_np    = np.load(os.path.join(basis_list[64], 'query_embed.c_exh.pth.npy'))

    net.input_proj.M_ui_np = np.load(os.path.join(basis_list[65], 'input_proj.M_ui_exh.pth.npy'))
    net.input_proj.c_np    = np.load(os.path.join(basis_list[65], 'input_proj.c_exh.pth.npy'))

    net.backbone[0].body.conv1.M_ui_np = np.load(os.path.join(basis_list[66], 'backbone.0.body.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.conv1.c_np    = np.load(os.path.join(basis_list[66], 'backbone.0.body.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer1[0].conv1.M_ui_np = np.load(os.path.join(basis_list[67], 'backbone.0.body.layer1.0.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[0].conv1.c_np    = np.load(os.path.join(basis_list[67], 'backbone.0.body.layer1.0.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer1[0].conv2.M_ui_np = np.load(os.path.join(basis_list[68], 'backbone.0.body.layer1.0.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[0].conv2.c_np    = np.load(os.path.join(basis_list[68], 'backbone.0.body.layer1.0.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer1[0].conv3.M_ui_np = np.load(os.path.join(basis_list[69], 'backbone.0.body.layer1.0.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[0].conv3.c_np    = np.load(os.path.join(basis_list[69], 'backbone.0.body.layer1.0.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer1[0].downsample[0].M_ui_np = np.load(os.path.join(basis_list[70], 'backbone.0.body.layer1.0.downsample.0.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[0].downsample[0].c_np    = np.load(os.path.join(basis_list[70], 'backbone.0.body.layer1.0.downsample.0.c_exh.pth.npy'))
    net.backbone[0].body.layer1[1].conv1.M_ui_np = np.load(os.path.join(basis_list[71], 'backbone.0.body.layer1.1.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[1].conv1.c_np    = np.load(os.path.join(basis_list[71], 'backbone.0.body.layer1.1.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer1[1].conv2.M_ui_np = np.load(os.path.join(basis_list[72], 'backbone.0.body.layer1.1.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[1].conv2.c_np    = np.load(os.path.join(basis_list[72], 'backbone.0.body.layer1.1.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer1[1].conv3.M_ui_np = np.load(os.path.join(basis_list[73], 'backbone.0.body.layer1.1.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[1].conv3.c_np    = np.load(os.path.join(basis_list[73], 'backbone.0.body.layer1.1.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer1[2].conv1.M_ui_np = np.load(os.path.join(basis_list[74], 'backbone.0.body.layer1.2.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[2].conv1.c_np    = np.load(os.path.join(basis_list[74], 'backbone.0.body.layer1.2.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer1[2].conv2.M_ui_np = np.load(os.path.join(basis_list[75], 'backbone.0.body.layer1.2.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[2].conv2.c_np    = np.load(os.path.join(basis_list[75], 'backbone.0.body.layer1.2.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer1[2].conv3.M_ui_np = np.load(os.path.join(basis_list[76], 'backbone.0.body.layer1.2.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer1[2].conv3.c_np    = np.load(os.path.join(basis_list[76], 'backbone.0.body.layer1.2.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer2[0].conv1.M_ui_np = np.load(os.path.join(basis_list[77], 'backbone.0.body.layer2.0.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[0].conv1.c_np    = np.load(os.path.join(basis_list[77], 'backbone.0.body.layer2.0.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer2[0].conv2.M_ui_np = np.load(os.path.join(basis_list[78], 'backbone.0.body.layer2.0.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[0].conv2.c_np    = np.load(os.path.join(basis_list[78], 'backbone.0.body.layer2.0.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer2[0].conv3.M_ui_np = np.load(os.path.join(basis_list[79], 'backbone.0.body.layer2.0.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[0].conv3.c_np    = np.load(os.path.join(basis_list[79], 'backbone.0.body.layer2.0.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer2[0].downsample[0].M_ui_np = np.load(os.path.join(basis_list[80], 'backbone.0.body.layer2.0.downsample.0.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[0].downsample[0].c_np    = np.load(os.path.join(basis_list[80], 'backbone.0.body.layer2.0.downsample.0.c_exh.pth.npy'))
    net.backbone[0].body.layer2[1].conv1.M_ui_np = np.load(os.path.join(basis_list[81], 'backbone.0.body.layer2.1.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[1].conv1.c_np    = np.load(os.path.join(basis_list[81], 'backbone.0.body.layer2.1.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer2[1].conv2.M_ui_np = np.load(os.path.join(basis_list[82], 'backbone.0.body.layer2.1.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[1].conv2.c_np    = np.load(os.path.join(basis_list[82], 'backbone.0.body.layer2.1.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer2[1].conv3.M_ui_np = np.load(os.path.join(basis_list[83], 'backbone.0.body.layer2.1.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[1].conv3.c_np    = np.load(os.path.join(basis_list[83], 'backbone.0.body.layer2.1.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer2[2].conv1.M_ui_np = np.load(os.path.join(basis_list[84], 'backbone.0.body.layer2.2.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[2].conv1.c_np    = np.load(os.path.join(basis_list[84], 'backbone.0.body.layer2.2.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer2[2].conv2.M_ui_np = np.load(os.path.join(basis_list[85], 'backbone.0.body.layer2.2.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[2].conv2.c_np    = np.load(os.path.join(basis_list[85], 'backbone.0.body.layer2.2.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer2[2].conv3.M_ui_np = np.load(os.path.join(basis_list[86], 'backbone.0.body.layer2.2.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[2].conv3.c_np    = np.load(os.path.join(basis_list[86], 'backbone.0.body.layer2.2.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer2[3].conv1.M_ui_np = np.load(os.path.join(basis_list[87], 'backbone.0.body.layer2.3.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[3].conv1.c_np    = np.load(os.path.join(basis_list[87], 'backbone.0.body.layer2.3.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer2[3].conv2.M_ui_np = np.load(os.path.join(basis_list[88], 'backbone.0.body.layer2.3.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[3].conv2.c_np    = np.load(os.path.join(basis_list[88], 'backbone.0.body.layer2.3.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer2[3].conv3.M_ui_np = np.load(os.path.join(basis_list[89], 'backbone.0.body.layer2.3.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer2[3].conv3.c_np    = np.load(os.path.join(basis_list[89], 'backbone.0.body.layer2.3.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer3[0].conv1.M_ui_np = np.load(os.path.join(basis_list[90], 'backbone.0.body.layer3.0.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[0].conv1.c_np    = np.load(os.path.join(basis_list[90], 'backbone.0.body.layer3.0.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer3[0].conv2.M_ui_np = np.load(os.path.join(basis_list[91], 'backbone.0.body.layer3.0.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[0].conv2.c_np    = np.load(os.path.join(basis_list[91], 'backbone.0.body.layer3.0.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer3[0].conv3.M_ui_np = np.load(os.path.join(basis_list[92], 'backbone.0.body.layer3.0.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[0].conv3.c_np    = np.load(os.path.join(basis_list[92], 'backbone.0.body.layer3.0.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer3[0].downsample[0].M_ui_np = np.load(os.path.join(basis_list[93], 'backbone.0.body.layer3.0.downsample.0.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[0].downsample[0].c_np    = np.load(os.path.join(basis_list[93], 'backbone.0.body.layer3.0.downsample.0.c_exh.pth.npy'))
    net.backbone[0].body.layer3[1].conv1.M_ui_np = np.load(os.path.join(basis_list[94], 'backbone.0.body.layer3.1.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[1].conv1.c_np    = np.load(os.path.join(basis_list[94], 'backbone.0.body.layer3.1.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer3[1].conv2.M_ui_np = np.load(os.path.join(basis_list[95], 'backbone.0.body.layer3.1.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[1].conv2.c_np    = np.load(os.path.join(basis_list[95], 'backbone.0.body.layer3.1.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer3[1].conv3.M_ui_np = np.load(os.path.join(basis_list[96], 'backbone.0.body.layer3.1.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[1].conv3.c_np    = np.load(os.path.join(basis_list[96], 'backbone.0.body.layer3.1.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer3[2].conv1.M_ui_np = np.load(os.path.join(basis_list[97], 'backbone.0.body.layer3.2.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[2].conv1.c_np    = np.load(os.path.join(basis_list[97], 'backbone.0.body.layer3.2.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer3[2].conv2.M_ui_np = np.load(os.path.join(basis_list[98], 'backbone.0.body.layer3.2.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[2].conv2.c_np    = np.load(os.path.join(basis_list[98], 'backbone.0.body.layer3.2.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer3[2].conv3.M_ui_np = np.load(os.path.join(basis_list[99], 'backbone.0.body.layer3.2.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[2].conv3.c_np    = np.load(os.path.join(basis_list[99], 'backbone.0.body.layer3.2.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer3[3].conv1.M_ui_np = np.load(os.path.join(basis_list[100], 'backbone.0.body.layer3.3.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[3].conv1.c_np    = np.load(os.path.join(basis_list[100], 'backbone.0.body.layer3.3.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer3[3].conv2.M_ui_np = np.load(os.path.join(basis_list[101], 'backbone.0.body.layer3.3.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[3].conv2.c_np    = np.load(os.path.join(basis_list[101], 'backbone.0.body.layer3.3.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer3[3].conv3.M_ui_np = np.load(os.path.join(basis_list[102], 'backbone.0.body.layer3.3.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[3].conv3.c_np    = np.load(os.path.join(basis_list[102], 'backbone.0.body.layer3.3.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer3[4].conv1.M_ui_np = np.load(os.path.join(basis_list[103], 'backbone.0.body.layer3.4.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[4].conv1.c_np    = np.load(os.path.join(basis_list[103], 'backbone.0.body.layer3.4.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer3[4].conv2.M_ui_np = np.load(os.path.join(basis_list[104], 'backbone.0.body.layer3.4.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[4].conv2.c_np    = np.load(os.path.join(basis_list[104], 'backbone.0.body.layer3.4.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer3[4].conv3.M_ui_np = np.load(os.path.join(basis_list[105], 'backbone.0.body.layer3.4.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[4].conv3.c_np    = np.load(os.path.join(basis_list[105], 'backbone.0.body.layer3.4.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer3[5].conv1.M_ui_np = np.load(os.path.join(basis_list[106], 'backbone.0.body.layer3.5.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[5].conv1.c_np    = np.load(os.path.join(basis_list[106], 'backbone.0.body.layer3.5.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer3[5].conv2.M_ui_np = np.load(os.path.join(basis_list[107], 'backbone.0.body.layer3.5.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[5].conv2.c_np    = np.load(os.path.join(basis_list[107], 'backbone.0.body.layer3.5.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer3[5].conv3.M_ui_np = np.load(os.path.join(basis_list[108], 'backbone.0.body.layer3.5.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer3[5].conv3.c_np    = np.load(os.path.join(basis_list[108], 'backbone.0.body.layer3.5.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer4[0].conv1.M_ui_np = np.load(os.path.join(basis_list[109], 'backbone.0.body.layer4.0.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[0].conv1.c_np    = np.load(os.path.join(basis_list[109], 'backbone.0.body.layer4.0.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer4[0].conv2.M_ui_np = np.load(os.path.join(basis_list[110], 'backbone.0.body.layer4.0.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[0].conv2.c_np    = np.load(os.path.join(basis_list[110], 'backbone.0.body.layer4.0.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer4[0].conv3.M_ui_np = np.load(os.path.join(basis_list[111], 'backbone.0.body.layer4.0.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[0].conv3.c_np    = np.load(os.path.join(basis_list[111], 'backbone.0.body.layer4.0.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer4[0].downsample[0].M_ui_np = np.load(os.path.join(basis_list[112], 'backbone.0.body.layer4.0.downsample.0.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[0].downsample[0].c_np    = np.load(os.path.join(basis_list[112], 'backbone.0.body.layer4.0.downsample.0.c_exh.pth.npy'))
    net.backbone[0].body.layer4[1].conv1.M_ui_np = np.load(os.path.join(basis_list[113], 'backbone.0.body.layer4.1.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[1].conv1.c_np    = np.load(os.path.join(basis_list[113], 'backbone.0.body.layer4.1.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer4[1].conv2.M_ui_np = np.load(os.path.join(basis_list[114], 'backbone.0.body.layer4.1.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[1].conv2.c_np    = np.load(os.path.join(basis_list[114], 'backbone.0.body.layer4.1.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer4[1].conv3.M_ui_np = np.load(os.path.join(basis_list[115], 'backbone.0.body.layer4.1.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[1].conv3.c_np    = np.load(os.path.join(basis_list[115], 'backbone.0.body.layer4.1.conv3.c_exh.pth.npy'))
    net.backbone[0].body.layer4[2].conv1.M_ui_np = np.load(os.path.join(basis_list[116], 'backbone.0.body.layer4.2.conv1.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[2].conv1.c_np    = np.load(os.path.join(basis_list[116], 'backbone.0.body.layer4.2.conv1.c_exh.pth.npy'))
    net.backbone[0].body.layer4[2].conv2.M_ui_np = np.load(os.path.join(basis_list[117], 'backbone.0.body.layer4.2.conv2.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[2].conv2.c_np    = np.load(os.path.join(basis_list[117], 'backbone.0.body.layer4.2.conv2.c_exh.pth.npy'))
    net.backbone[0].body.layer4[2].conv3.M_ui_np = np.load(os.path.join(basis_list[118], 'backbone.0.body.layer4.2.conv3.M_ui_exh.pth.npy'))
    net.backbone[0].body.layer4[2].conv3.c_np    = np.load(os.path.join(basis_list[118], 'backbone.0.body.layer4.2.conv3.c_exh.pth.npy'))
    
    net.class_embed.weight_approx_np = np.load(os.path.join(basis_list[118], 'class_embed.weight_approx.npy'))
    
    layer_info = []
    cnt = 0
    for name, param in net.state_dict().items():
        if '.' not in name:
            continue
        layer_split = name.split('.')
        name_len = len(layer_split)
        _, param_name_re = name.rsplit('.', 1)
         
        if name_len == 2:
            layer_name, param_name = layer_split
            layer_rename = layer_name
            layer_param_name = name
        elif name_len == 4:
            conv_name, conv_idx, func_name, param_name = layer_split
            layer_rename = '{}.{}[{}]'.format(conv_name, conv_idx, func_name)
            layer_param_name = layer_rename + '.' + param_name
        elif name_len == 5:
            conv_name, conv_idx, func_name, func_idx, param_name = layer_split
            if conv_idx == '0':
                layer_rename = '{}[{}].{}.{}'.format(conv_name, conv_idx, func_name, func_idx)
            else:
                layer_rename = '{}.{}.{}.{}'.format(conv_name, conv_idx, func_name, func_idx)
            layer_param_name = layer_rename + '.' + param_name
        elif name_len == 6:
            conv_name, conv_idx, func_name, func_idx, option_name, param_name = layer_split
            layer_rename = '{}.{}.{}[{}].{}'.format(conv_name, conv_idx, func_name, func_idx, option_name)
            layer_param_name = layer_rename + '.' + param_name
        elif name_len == 7:
            name_a, name_b, name_c, name_d, name_e, name_f, param_name =layer_split
            if name_b == '0':
                layer_rename = '{}[{}].{}.{}[{}].{}'.format(name_a, name_b, name_c, name_d, name_e, name_f)
            else:
                layer_rename = '{}.{}.{}[{}].{}.{}'.format(name_a, name_b, name_c, name_d, name_e, name_f)
            layer_param_name = layer_rename + '.' + param_name
        elif name_len == 8:
            name_a, name_b, name_c, name_d, name_e, name_f, name_g, param_name =layer_split
            layer_rename = '{}[{}].{}.{}[{}].{}[{}]'.format(name_a, name_b, name_c, name_d, name_e, name_f, name_g)
            layer_param_name = layer_rename + '.' + param_name        

        if param_name_re == 'weight' or param_name_re == 'in_proj_weight':
            weight_dim = net.state_dict()[name].dim()
            if (weight_dim == 2) or (weight_dim == 4):
                if text2list(param_info_path[cnt], name) != []:
                    d_param = float(text2list(param_info_path[cnt], name)[0][3])
                    layer_info.append([layer_rename, Q_bits_list[cnt], layer_basis[cnt], d_param])
                    print('{}\tquantize_bits={}  basis={}  d_param={:.4f}[MB]'.format(
                        layer_rename, Q_bits_list[cnt], layer_basis[cnt], d_param))
                    cnt += 1
  
        if ('M_ui' == param_name_re) or ('c' == param_name_re):
            exec("net.{}.basis = layer_basis[cnt-1]".format(layer_rename))
            exec("net.{}.quantize_bits = Q_bits_list[cnt - 1]".format(layer_rename))
            exec("net.{}.num_threads = {}".format(layer_rename, num_threads))

        if 'weight_approx' == param_name_re:
            new_param = np.load(os.path.join(basis_list[cnt-1], name + '.npy')).astype(np.float32)
            new_param = Parameter(torch.from_numpy(new_param), requires_grad=False)
            exec("net.{} = new_param".format(layer_param_name))

    print(f'+---{cnt} parameters loaded---+')

    calc_params(net, layer_info, total_param)



def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    device = torch.device(args.device)
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)['model'], strict=False) 

    range_q_bits_all = [args.qb for i in range(119)]
    range_basis_all  = [args.qb for i in range(119)]
    

    dir_param = os.path.join(os.path.dirname(args.weights), args.binary_weights)
    print(dir_param)
    load_param(model, range_q_bits_all, range_basis_all, dir_param, d_mode='exh')

    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)
    output_dir = Path(args.output_dir)

    if args.eval:
        coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
