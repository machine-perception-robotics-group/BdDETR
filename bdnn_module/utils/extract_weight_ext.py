import argparse
import csv
import os
import gc
import time
import torch
from torch.nn.parameter import Parameter
import numpy as np
from bdnn_module.utils import decomposition as pp

def calc_param_size(weight, dtype, decomposed=False):
    weight_shape = weight.shape
    if decomposed:
        I, O = weight_shape
        param_num = (I * O * dtype) / (8 * 1024 ^ 2)
    else:
        if len(weight_shape) == 4:
            N, C, H, W = weight_shape
            param_num = (N * C * H * W * dtype) / (8 * 1024 ^ 2)
        elif len(weight_shape) == 2:
            I, O = weight_shape
            param_num = (I * O * dtype) / (8 * 1024 ^ 2)
        else:
            param_num = None
            print('weight shape must be 4 or 2 dimension.')

    return param_num

def save_list(s_list, w_mode, save_dir):
    with open(save_dir, w_mode) as f:
        dataWriter = csv.writer(f, lineterminator='\n')
        dataWriter.writerows(s_list)

def save_list_1rows(s_list, w_mode, save_dir):
    s_list_ = [str(i) for i in s_list]
    result_list_ = '\n'.join(s_list_)
    with open(save_dir, w_mode) as f:
        f.write(result_list_)
        f.write('\n\n')

def save_param(model, basis, mode, save_dir, w_param=False):
    model.eval()
    dim_len = 64

    print('target basis list: ' + str(basis))
    for k in basis:
        print('\n-------- decomposition basis: {} --------'.format(k))
        print('     layer               param      decomposed    8bit    compress      restore     time')
        save_basis_path = os.path.join(save_dir, 'B' + str(k))
        os.makedirs(save_basis_path, exist_ok=True)
        layer_conv = 0
        layer_fc = 0
        layer_rnn = 0
        list_param_info = []
        decompose_time_list = []
        w_result = []
        d_result = []
        
        for name, param in model.state_dict().items():
            t1 = time.time()
            save_layer_path = os.path.join(save_basis_path, name.replace('.weight', ''))
            weight_dim = model.state_dict()[name].dim()
            weight_shape = np.array(model.state_dict()[name].shape, dtype=np.int32)
            weight_np = model.state_dict()[name].cpu().clone().detach().numpy()
            weight_np = weight_np.astype(np.float64)
            rnn_flag = False
            
            if 'weight' in name:
                if weight_dim == 4:
                    in_size = weight_shape[1] * weight_shape[2] * weight_shape[3]
                    layer_conv += 1
                elif weight_dim == 2:
                    if 'weight_ih' in name:
                        layer_rnn += 1
                        rnn_flag = True
                    elif 'weight_hh' in name:
                        rnn_flag = True
                        pass
                    else:
                        layer_fc += 1
                    in_size = weight_shape[1]
                else:
                    continue

                if rnn_flag:
                    os.makedirs(save_layer_path, exist_ok=True)

                    w_ih_shape = weight_np.shape
                    w_ih_sep = w_ih_shape[0] // 4
                    w_ih_sep = [w_ih_sep, 2 * w_ih_sep, 3 * w_ih_sep, w_ih_shape[0]]
                    w_ih_sep_len = w_ih_sep[0]
                    bitset = (in_size + (dim_len - 1)) // dim_len
                    M_ui_np = np.zeros((weight_shape[0], k, bitset), dtype=np.uint64)
                    c_np = np.zeros((weight_shape[0], k), dtype=np.float64)
                    weight_approx = np.zeros(weight_shape, dtype=np.float32)
                    M_np = np.zeros((weight_shape[0], weight_shape[1]*k), dtype=np.int64)
                    restore_ratio_ih = np.zeros(4, dtype=np.float32)

                    w_info = np.zeros((weight_shape[0], 12), dtype=np.float64)
                    M_ii, c_ii_np, weight_approx_ii_np, restore_ratio_ih[0], w_info[:w_ih_sep[0]] = pp.decomposition_check(weight_np[:w_ih_sep[0], :], k, mode)
                    M_if, c_if_np, weight_approx_if_np, restore_ratio_ih[1], w_info[w_ih_sep[0]:w_ih_sep[1]] = pp.decomposition_check(weight_np[w_ih_sep[0]:w_ih_sep[1], :], k, mode)
                    M_ig, c_ig_np, weight_approx_ig_np, restore_ratio_ih[2], w_info[w_ih_sep[1]:w_ih_sep[2]] = pp.decomposition_check(weight_np[w_ih_sep[1]:w_ih_sep[2], :], k, mode)
                    M_io, c_io_np, weight_approx_io_np, restore_ratio_ih[3], w_info[w_ih_sep[2]:] = pp.decomposition_check(weight_np[w_ih_sep[2]:, :], k, mode)

                    M_np[:w_ih_sep[0]] = M_ii
                    M_np[w_ih_sep[0]:w_ih_sep[1]] = M_if
                    M_np[w_ih_sep[1]:w_ih_sep[2]] = M_ig
                    M_np[w_ih_sep[2]:] = M_io

                    M_ui_ii_np = pp.compression(in_size, w_ih_sep_len, k, dim_len, bitset, M_ii)
                    M_ui_if_np = pp.compression(in_size, w_ih_sep_len, k, dim_len, bitset, M_if)
                    M_ui_ig_np = pp.compression(in_size, w_ih_sep_len, k, dim_len, bitset, M_ig)
                    M_ui_io_np = pp.compression(in_size, w_ih_sep_len, k, dim_len, bitset, M_io)

                    M_ui_np[:w_ih_sep[0]] = M_ui_ii_np
                    M_ui_np[w_ih_sep[0]:w_ih_sep[1]] = M_ui_if_np
                    M_ui_np[w_ih_sep[1]:w_ih_sep[2]] = M_ui_ig_np
                    M_ui_np[w_ih_sep[2]:] = M_ui_io_np

                    c_np[:w_ih_sep[0]] = c_ii_np
                    c_np[w_ih_sep[0]:w_ih_sep[1]] = c_if_np
                    c_np[w_ih_sep[1]:w_ih_sep[2]] = c_ig_np
                    c_np[w_ih_sep[2]:] = c_io_np

                    weight_approx[:w_ih_sep[0], :] = weight_approx_ii_np
                    weight_approx[w_ih_sep[0]:w_ih_sep[1], :] = weight_approx_if_np
                    weight_approx[w_ih_sep[1]:w_ih_sep[2], :] = weight_approx_ig_np
                    weight_approx[w_ih_sep[2]:, :] = weight_approx_io_np

                    restore_ratio = np.mean(restore_ratio_ih)

                    np.save(os.path.join(save_layer_path, 'M_ui_ig_' + mode), M_ui_ig_np)
                    np.save(os.path.join(save_layer_path, 'M_ui_ii_' + mode), M_ui_ii_np)
                    np.save(os.path.join(save_layer_path, 'M_ui_io_' + mode), M_ui_io_np)
                    np.save(os.path.join(save_layer_path, 'M_ui_if_' + mode), M_ui_if_np)
                    np.save(os.path.join(save_layer_path, 'c_ig_' + mode), c_ig_np)
                    np.save(os.path.join(save_layer_path, 'c_ii_' + mode), c_ii_np)
                    np.save(os.path.join(save_layer_path, 'c_io_' + mode), c_io_np)
                    np.save(os.path.join(save_layer_path, 'c_if_' + mode), c_if_np)

                else:
                    bitset = (in_size + (dim_len - 1)) // dim_len
                    M_np, c_np, weight_approx, restore_ratio, w_info = pp.decomposition_check(weight_np, k, mode)
                    M_ui_np = pp.compression(in_size, weight_shape[0], k, dim_len, bitset, M_np)

                    np.save(save_layer_path + '.M_ui_' + mode + '.pth', M_ui_np)
                    np.save(save_layer_path + '.c_' + mode + '.pth', c_np)

                    if w_param:
                        np.save(save_layer_path + '.weight.npy', weight_np)
                        np.save(save_layer_path + '.weight_approx.npy', weight_approx)

                np.savetxt(save_layer_path + '.info' + mode + '.csv', w_info, delimiter=',')

                d_time = time.time() - t1
                decompose_time_list.append(d_time)
                abs_approx = np.abs(weight_np - weight_approx)
                w_result.append([name, restore_ratio, np.mean(weight_np), np.median(weight_np), np.std(weight_np,
                                                                                                       ddof=1)])
                d_result.append([name, restore_ratio, np.mean(weight_approx), np.median(weight_approx),
                                 np.std(weight_approx, ddof=1), np.sum(abs_approx),
                                 np.mean(abs_approx)])
                param_size = calc_param_size(weight_np, 32) / 1024.0
                M_size, c_size = calc_param_size(M_np, 1, True), calc_param_size(c_np, 64, True)
                d_param_size = (M_size + c_size) / 1024.0
                param_size_8bit = calc_param_size(weight_np, 8) / 1024.0
                compress_ratio = d_param_size / param_size
                list_param_info.append([name, restore_ratio, param_size, d_param_size, compress_ratio, param_size_8bit])
                print('  {}\t{:.4f}[MB]    {:.4f}[MB]   {:.4f}[MB]   {:.4f}[%]   {:.4f}[%]   {:.4f}[sec]   {}'.format(
                    name.replace('.weight', ''), param_size, d_param_size, param_size_8bit, 100 * (1.0 - compress_ratio), restore_ratio, d_time, weight_shape))

                del M_np
                del M_ui_np
                del c_np
                del weight_np
                del weight_approx
                gc.collect()

        decompose_time_list.append(sum(decompose_time_list))
        ave_restore_ratio = sum([x[1] for x in list_param_info]) / len(list_param_info)
        all_param_size = sum([x[2] for x in list_param_info])
        all_d_param_size = sum([x[3] for x in list_param_info])
        all_8bit_size = sum([x[5] for x in list_param_info])
        all_compressing_ratio = all_d_param_size / all_param_size
        list_param_info.append(['ALL', ave_restore_ratio, all_param_size, all_d_param_size, all_compressing_ratio, all_8bit_size])
        save_list(list_param_info, 'a', os.path.join(save_basis_path, 'param_info_' + mode + '.csv'))
        save_list(w_result, 'a', os.path.join(save_basis_path, 'weight_info_' + mode + '.csv'))
        save_list(d_result, 'a', os.path.join(save_basis_path, 'approx_info_' + mode + '.csv'))
        save_list_1rows(decompose_time_list, 'a', os.path.join(save_basis_path, 'decompose_time' + mode + '.csv'))

        print('conv:{}  fc:{}  rnn:{}  param:{:.4f}[MB]  decomposed:{:.4f}[MB]  8bit:{:.4f}[MB]  compress:{:.2f}[%]'.format(
            layer_conv, layer_fc, layer_rnn, all_param_size, all_d_param_size, all_8bit_size, 100 * (1.0 - all_compressing_ratio)))

