import numpy as np
import numba as nb
import time

def decomposition(weight_np, basis, mode):
    out_channel, in_channel = weight_np.shape[:2]
    before_shape = weight_np.shape

    if weight_np.ndim == 4:
        conv_layer = True
        kh, kw = weight_np.shape[2:]
        in_channel = in_channel * kh * kw
        weight_np = weight_np.reshape((out_channel, in_channel))
        weight_approx = np.zeros((out_channel, in_channel))
    else:
        conv_layer = False
        weight_approx = np.zeros((out_channel, in_channel))

    if mode == 'exh':
        from bdnn_module.Exhaustive_decomposer_numpy import Exhaustive_decompose as decomposer
        base = 2
        ranges = base ** basis
        all_candidates_list = [list(map(int, format(i, 'b').zfill(basis))) for i in range(ranges)]
        all_candidates = np.array(all_candidates_list, dtype=np.float64)
        all_candidates = all_candidates * 2 - 1
    else:
        from bdnn_module.Greedy_decomposer import Greedy_decompose as decomposer
        all_candidates = None

    M = np.zeros((out_channel, in_channel * basis), dtype=np.float64)
    c = np.zeros((out_channel, basis), dtype=np.float64)
    restore_ratio = 0.0

    for i in range(out_channel):
        mat_t, c_t = decomposer(weight_np[i, :], basis, all_candidates)
        weight_approx[i, :] = np.dot(mat_t, c_t)
        M[i, :] = np.ravel(mat_t, order='F')
        c[i, :] = c_t
        abs_approx = np.abs(weight_np[i, :] - np.abs(weight_np[i, :] - weight_approx[i, :]))
        abs_weight = np.abs(weight_np[i, :])
        restore_ratio = restore_ratio + np.sum(np.nan_to_num(np.minimum(abs_approx, abs_weight)
                                                             / np.maximum(abs_approx, abs_weight)))

    if conv_layer:
        weight_approx = weight_approx.reshape(before_shape)
    restore_ratio = 100 * restore_ratio / weight_np.size
    M = np.where(M == 1.0, 1.0, 0.0)
    M = M.astype(np.int64)

    return M, c, weight_approx, restore_ratio

def decomposition_check(weight_np, basis, mode):
    out_channel, in_channel = weight_np.shape[:2]
    before_shape = weight_np.shape

    if weight_np.ndim == 4:
        conv_layer = True
        kh, kw = weight_np.shape[2:]
        in_channel = in_channel * kh * kw
        weight_np = weight_np.reshape((out_channel, in_channel))
        weight_approx = np.zeros((out_channel, in_channel))
    else:
        conv_layer = False
        weight_approx = np.zeros((out_channel, in_channel))

    if mode == 'exh':
        from bdnn_module.Exhaustive_decomposer_numpy import Exhaustive_decompose as decomposer
        base = 2
        ranges = base ** basis
        all_candidates_list = [list(map(int, format(i, 'b').zfill(basis))) for i in range(ranges)]
        all_candidates = np.array(all_candidates_list, dtype=np.float64)
        all_candidates = all_candidates * 2 - 1
        all_candidates = np.ascontiguousarray(all_candidates, dtype=np.float64)
    else:
        from bdnn_module.Greedy_decomposer import Greedy_decompose as decomposer
        all_candidates = None

    M = np.zeros((out_channel, in_channel * basis), dtype=np.float64)
    c = np.zeros((out_channel, basis), dtype=np.float64)
    w_info = np.zeros((out_channel, 12), dtype=np.float64)
    restore_ratio = 0.0
    
    for i in range(out_channel):
        weight_vec = np.ascontiguousarray(weight_np[i, :])
        mat_t, c_t = decomposer(weight_vec, basis, all_candidates)
        weight_approx[i, :] = np.dot(mat_t, c_t)
        M[i, :] = np.ravel(mat_t, order='F')
        c[i, :] = c_t
        w_info[i, :] = calc_info(weight_np[i, :], weight_approx[i, :])
        abs_approx = np.abs(weight_np[i, :] - np.abs(weight_np[i, :] - weight_approx[i, :]))
        abs_weight = np.abs(weight_np[i, :])
        restore_ratio = restore_ratio + np.sum(np.nan_to_num(np.minimum(abs_approx, abs_weight)
                                                             / np.maximum(abs_approx, abs_weight)))

    if conv_layer:
        weight_approx = weight_approx.reshape(before_shape)
    restore_ratio = 100 * restore_ratio / weight_np.size
    M = np.where(M == 1.0, 1.0, 0.0)
    M = M.astype(np.int64)

    return M, c, weight_approx, restore_ratio, w_info


@nb.jit(nb.float64(nb.float64[:]), nopython=True, nogil=True)
def mode1(a):
    n, bins = np.histogram(a)
    idx = np.argmax(n)
    x_mode = (bins[idx] + bins[idx+1]) / 2
    return x_mode


@nb.jit(nb.float64(nb.float64[:], nb.float64[:]), nopython=True, nogil=True)
def entropy_multi(a, b):
    return np.sum(a * np.log(a / b), axis=0)


@nb.jit(nb.float64(nb.float64[:]), nopython=True, nogil=True)
def entropy_single(a):
    return np.sum(a * np.log(a), axis=0)


@nb.jit(nb.float64(nb.float64[:], nb.float64[:]), nopython=True, nogil=True)
def KLD(a, b):
    a = 1.0 * a / np.sum(a, axis=0)
    b = 1.0 * b / np.sum(b, axis=0)
    return entropy_multi(a, b)

def calc_info(w1, w2):
    w_info = np.zeros(12, dtype=np.float64)
    w_info[0] = np.mean(w1)
    w_info[1] = np.mean(w2)
    w_info[2] = np.median(w1)
    w_info[3] = np.median(w2)
    w_info[4] = mode1(w1)
    w_info[5] = mode1(w2)
    w_info[6] = np.std(w1)
    w_info[7] = np.std(w2)
    w_info[8] = np.mean(np.fft.fft(w1))
    w_info[9] = np.mean(np.fft.fft(w2))
    w_info[10] = np.sum(np.abs(w1 - w2))
    w_info[11] = KLD(w1, w2)

    return w_info

@nb.jit(nb.uint64[:, :, :](nb.int64, nb.int64, nb.int64, nb.int64, nb.int64, nb.int64[:, :]),
        nopython=True, nogil=True, parallel=True)
def compression(in_size, w_out, basis, dim_len, bitset, M):
    M_ui = np.zeros((w_out, basis, bitset), dtype=np.uint64)

    for d1 in range(w_out):
        for k in range(basis):
            for dim in range(in_size):
                dim_ui = dim // dim_len
                M_ui[d1, k, dim_ui] <<= np.uint64(1)
                if 1 == M[d1, (in_size*k)+dim]:
                    M_ui[d1, k, dim_ui] += 1

    return M_ui


@nb.jit(nb.float64[:](nb.int64, nb.float64[:, :, :, :]), nopython=True, nogil=True)
def offset(in_size, weight):
    offset_vec = np.zeros(weight.shape[0], dtype=np.float64)
    weight_flatten = weight.ravel()
    for w1 in range(weight.shape[0]):
        begin = w1 * in_size
        end = begin + in_size
        offset_vec[w1] = np.sum(weight_flatten[begin:end])

    return offset_vec


@nb.jit(nb.float64[:](nb.int64, nb.float64[:, :]), nopython=True, nogil=True)
def offset_fc(in_size, weight):
    offset_vec = np.zeros(weight.shape[0], dtype=np.float64)
    weight_flatten = weight.ravel()
    for w1 in range(weight.shape[0]):
        begin = w1 * in_size
        end = begin + in_size
        offset_vec[w1] = np.sum(weight_flatten[begin:end])

    return offset_vec