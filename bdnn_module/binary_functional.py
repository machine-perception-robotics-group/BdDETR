from traceback import print_tb
import torch
import time
import multiprocessing as mp
import numpy as np
import numba as nb
import torch.nn as nn
from torch.nn import functional as F
from torch._C import _infer_size, _add_docstr
from torch.nn import grad
from torch._jit_internal import List

from bdnn_module import binaryfunc_cython as bfc
from bdnn_module.utils import decomposition as pp
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def is_int_or_list(v):
    return type(v) in (int, list)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def quantization(x, s, z, alpha_q, beta_q):
    x_q = np.round(1 / s * x + z, decimals=0)
    x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)
    return x_q


def quantization_int8(x, s, z):
    x_q = quantization(x, s, z, alpha_q=-128, beta_q=127)
    x_q = x_q.astype(np.int8)
    return x_q


def dequantization(x_q, s, z):
    x_q = x_q.astype(np.int32)
    x = s * (x_q - z)
    x = x.astype(np.float32)
    return x


def generate_quantization_constants(alpha, beta, alpha_q, beta_q):
    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))
    return s, z


def generate_quantization_int8_constants(alpha, beta):
    b = 8
    alpha_q = -2**(b - 1)
    beta_q = 2**(b - 1) - 1
    s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q)
    return s, z

def quantization_matrix_multiplication_int8(X_q, W_q, b_q, s_X, z_X, s_W, z_W,
                                            s_b, z_b, s_Y, z_Y):
    p = W_q.shape[0]
    Y_q_simulated = (z_Y + (s_b / s_Y * (b_q.astype(np.int32) - z_b)) + (
        (s_X * s_W / s_Y) *
        (np.matmul(X_q.astype(np.int32), W_q.astype(np.int32)) -
         z_W * np.sum(X_q.astype(np.int32), axis=1, keepdims=True) - z_X *
         np.sum(W_q.astype(np.int32), axis=0, keepdims=True) + p * z_X * z_W)))

    Y_q_simulated = np.round(Y_q_simulated, decimals=0)
    Y_q_simulated = np.clip(Y_q_simulated, a_min=-128, a_max=127)
    Y_q_simulated = Y_q_simulated.astype(np.int8)

    return Y_q_simulated

# conv2d_float (DETR)
# def conv2d_binary(input, weight_np, offset_W, M_ui_np, c_np, stride=1, padding=0, dilation=1, groups=1, quantize_bits=1,
#                   basis=1, train_b=True, train_epoch=True, mode='exh', num_threads=4, bias=None):

#     dim_len = 64
#     in_shape = input.shape
#     w_shape = weight_np.shape
#     w_shape = np.array(w_shape, dtype=np.int32)
#     weight_approx = weight_np
#     sH, sW = stride
#     pH, pW = padding
#     dH, dW = dilation

#     if input.dim() == 4:
#         input_np = input.cpu().clone().detach().numpy()
#         input_np = input_np.astype(np.float64)

#         if not train_b:
#             if c_np[0, 0] == 0:
#                 weight = torch.as_tensor(weight_np, dtype=torch.float32, device='cpu')
#                 ret = F.conv2d(input, weight, bias, stride, padding, dilation)
#                 ret_np = ret.cpu().clone().detach().numpy()
#             else:
#                 ret_np = bfc.conv2d_float_popcnt(input_np, weight_np, M_ui_np, c_np, sH, sW, pH, pW, dH, dW, groups, quantize_bits, basis, dim_len, num_threads)

#         weight_approx = torch.as_tensor(weight_approx, dtype=torch.float32, device='cpu')
#         ret = torch.as_tensor(ret_np, dtype=torch.float32, device='cpu')
#     else:
#         ret = input
#         print('unsupported_format')
#         exit(1)

#     if bias is not None:
#         ret = bias.view(-1, 1, 1) + ret

#     return ret, weight_approx, M_ui_np, c_np


# BdDETR
def conv2d_binary(input, weight_np, offset_W, M_ui_np, c_np, stride=1, padding=0, dilation=1, groups=1, quantize_bits=1,
                  basis=1, train_b=True, train_epoch=True, mode='exh', num_threads=4, bias=None):
    dim_len = 64
    in_shape = input.shape
    w_shape = weight_np.shape
    w_shape = np.array(w_shape, dtype=np.int32)
    weight_approx = weight_np
    sH, sW = stride
    pH, pW = padding
    dH, dW = dilation

    if input.dim() == 4:
        in_size = in_shape[1] * w_shape[2] * w_shape[3]
        bitset = (in_size + (dim_len - 1)) // dim_len
        input_np = input.cpu().clone().detach().numpy()
        input_np = input_np.astype(np.float64)
        min_val = np.min(input_np)

        if not train_b:
            if c_np[0, 0] == 0:
                weight = torch.as_tensor(weight_np, dtype=torch.float32, device='cpu')
                ret = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
                ret_np = ret.cpu().clone().detach().numpy()

            if min_val < 0.0:
                ret_np = bfc.conv2d_binary_popcnt_offset(input_np, w_shape, M_ui_np, c_np, offset_W, sH, sW, pH, pW, dH, dW,
                                                         groups, quantize_bits, basis, dim_len, num_threads)
            else:
                ret_np = bfc.conv2d_binary_popcnt(input_np, w_shape, M_ui_np, c_np, sH, sW, pH, pW, dH, dW, groups,
                                                  quantize_bits, basis, dim_len, num_threads)

        else:
            if not train_epoch:
                ret = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
                ret_np = ret.cpu().clone().detach().numpy()
                weight_approx = weight_np
            else:
                M, c, weight_approx, _ = pp.decomposition(weight_np, basis, mode)
                M_ui = pp.compression(in_size, w_shape[0], basis, dim_len, bitset, M)
                if min_val < 0.0:
                    ret_np = bfc.conv2d_binary_popcnt_offset(input_np, w_shape, M_ui_np, c_np, offset_W, sH, sW, pH, pW,
                                                             dH, dW, groups, quantize_bits, basis, dim_len, num_threads)
                else:
                    ret_np = bfc.conv2d_binary_popcnt(input_np, w_shape, M_ui_np, c_np, sH, sW, pH, pW, dH, dW, groups,
                                                      quantize_bits, basis, dim_len, num_threads)
                M_ui = torch.as_tensor(M_ui.astype(np.int64), dtype=torch.int64, device='cpu')
                c = torch.as_tensor(c, dtype=torch.float64, device='cpu')
                M_ui.requires_grad = False
                c.requires_grad = False

        weight_approx = torch.as_tensor(weight_approx, dtype=torch.float32, device='cpu')
        ret = torch.as_tensor(ret_np, dtype=torch.float32, device='cpu')
    else:
        ret = input
        print('unsupported_format')
        exit(1)

    if bias is not None:
        ret = bias.view(-1, 1, 1) + ret

    return ret, weight_approx, M_ui_np, c_np

# linear_float (DETR)
# def linear_binary_KA(input, weight, weight_approx, offset_w, M_ui_np, c_np, quantize_bits, basis, train_b, train_epoch, mode, num_threads=4, bias=None):

#     t1 = time.time()
#     dim_len: int = 64
#     in_shape = input.shape
#     w_shape = weight.shape

#     input_np = input.cpu().clone().detach().numpy()
#     input_np = input_np.astype(np.float64)
#     weight_np = weight.cpu().clone().detach().numpy()
#     weight_np = weight_np.astype(np.float64)
#     input_dim = input.dim()

#     ret_np = 0
#     if c_np[0, 0] == 0.0:
#         if input.shape[0] == weight.T.shape[0]:
#             ret = torch.matmul(input, weight.T)
#         else:
#             ret = torch.matmul(input, weight.T)
#         if bias is not None:
#             ret += bias
#         return ret, weight_approx, M_ui_np, c_np

#     if input_dim == 2:
#         ret_np = bfc.linear_binary_popcnt_2axis(input_np, w_shape[0], M_ui_np, c_np, offset_w, quantize_bits, basis, dim_len, num_threads)
#     elif input_dim == 3:
#         ret_np = bfc.linear_binary_popcnt_3float(input_np, weight_np, M_ui_np, c_np, offset_w, quantize_bits, basis, dim_len, num_threads)
#     elif input_dim == 4:
#         ret_np = bfc.linear_binary_popcnt_4float(input_np, weight_np, M_ui_np, c_np, offset_w, quantize_bits, basis, dim_len, num_threads)
#     else:
#         print(f'[Linear] unsupported type:{in_shape}')
#         exit()

#     ret = torch.as_tensor(ret_np, dtype=torch.float32, device=torch.device('cpu'))
#     if bias is not None:
#         ret += bias
    
#     return ret, weight_approx, M_ui_np, c_np

# BdDETR
def linear_binary_KA(input, weight, weight_approx, offset_w, M_ui_np, c_np, quantize_bits, basis, train_b, train_epoch, mode, num_threads=4, bias=None):
    t1 = time.time()
    dim_len: int = 64
    in_shape = input.shape
    w_shape = weight.shape

    input_np = input.cpu().clone().detach().numpy()
    input_np = input_np.astype(np.float64)
    input_dim = input.dim()

    ret_np = 0
    if c_np[0, 0] == 0.0:
        if input.shape[0] == weight.T.shape[0]:
            ret = torch.matmul(input, weight.T)
        else:
            ret = torch.matmul(input, weight.T)
        if bias is not None:
            ret += bias
        return ret, weight_approx, M_ui_np, c_np

    if input_dim == 2:
        ret_np = bfc.linear_binary_popcnt_2axis(input_np, w_shape[0], M_ui_np, c_np, offset_w, quantize_bits, basis, dim_len, num_threads)
    elif input_dim == 3:
        ret_np = bfc.linear_binary_popcnt_3axis(input_np, w_shape[0], M_ui_np, c_np, offset_w, quantize_bits, basis, dim_len, num_threads)
    elif input_dim == 4:
        ret_np = bfc.linear_binary_popcnt_4axis(input_np, w_shape[0], M_ui_np, c_np, offset_w, quantize_bits, basis, dim_len, num_threads)
    else:
        print(f'[Linear] unsupported type:{in_shape}')
        exit()

    ret = torch.as_tensor(ret_np, dtype=torch.float32, device=torch.device('cpu'))
    if bias is not None:
        ret += bias

    return ret, weight_approx, M_ui_np, c_np

