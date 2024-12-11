#cython: language_level=3
#cython: boundscheck(False)
#cython: wraparound(False)
#cython: nonecheck(False)
#cython: cdivision(True)

import numpy as np
cimport cython
cimport numpy as np
from cython.parallel cimport parallel, prange
cimport openmp

from libc.math cimport pow
from libc.stdint cimport int64_t, uint64_t

I32 = np.int32
F64 = np.float64
I64 = np.int64
UI64 = np.uint64
ctypedef np.int32_t I32_t
ctypedef np.float64_t F64_t
ctypedef np.int64_t I64_t
ctypedef np.uint64_t UI64_t

from libc.math cimport nan
from libc.math cimport isnan

# import popcnt
cdef extern from "nmmintrin.h":
    int64_t _mm_popcnt_u64 (uint64_t) nogil


"""
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[:, :, :, :] conv2d_float_before_product(np.ndarray[F64_t, ndim=4] input_np, np.ndarray[F64_t, ndim=4] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    cdef int batch1, batch2 = 0
    cdef int in_shape_d1 = input_np.shape[0], in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2], in_shape_d4 = input_np.shape[3]
    cdef int w_shape_d1 = weight_np.shape[0], w_shape_d2 = weight_np.shape[1]
    cdef int w_shape_d3 = weight_np.shape[2], w_shape_d4 = weight_np.shape[3]
    cdef int kh = w_shape_d3, kw = w_shape_d4
    cdef int out_shape_d1 = in_shape_d1, out_shape_d2 = w_shape_d1
    cdef int out_shape_d3 = <int>((in_shape_d3 - kh + (2 * padding_h)) / stride_h + 1)
    cdef int out_shape_d4 = <int>((in_shape_d4 - kw + (2 * padding_w)) / stride_w + 1)
    
    cdef double[:, :, :, :] ret = np.zeros([out_shape_d1, out_shape_d2, out_shape_d3, out_shape_d4], dtype=F64)

    # padding
    cdef np.ndarray[F64_t, ndim=4] pad_src = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3+(2*padding_h),\
                                                       in_shape_d4+(2*padding_w)], dtype=F64)
    if  1 <= padding_h or 1 <= padding_w:
        for batch1 in range(in_shape_d1):
            for ch in range(in_shape_d2):
                for height in range(in_shape_d3):
                    for width in range(in_shape_d4):
                        pad_h = padding_h + height
                        pad_w = padding_w + width
                        pad_src[batch1, ch, pad_h, pad_w] = input_np[batch1, ch, height, width]
    else:
        pad_src = input_np

    # convolution
    cdef int och = 0, ich = 0, i = 0, j = 0, kheight = 0, kwidth = 0

    if dilation_h == 1 and dilation_w == 1:
        pass
    else:
        print("This parameter is not supported")
        exit(1)

    with nogil, parallel(num_threads=1):
    #with nogil, parallel(num_threads=num_threads):
        for batch2 in range(in_shape_d1):
            for och in range(w_shape_d1):
                for ich in range(in_shape_d2):
                    for i in range(out_shape_d3): # 0-13
                        for j in range(out_shape_d4): # 0-13
                            for kheight in range(w_shape_d3): # 0-15
                                for kwidth in range(w_shape_d4): # 0-15
                                    ret[batch2, och, i, j] += pad_src[batch2, ich, kheight + (i * stride_h), kwidth + (j * stride_w)] * weight_np[och, ich, kheight, kwidth]

    return ret
"""


@cython.boundscheck(False)  # overflow/underflow check off
@cython.wraparound(False)  #  negative index check off
@cython.nonecheck(False)  # none check off
cdef double[:, :] linear_binary_product(np.ndarray[F64_t, ndim=2] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    
    cdef int batch, dim, dim_ui, exp
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef long bitset = <long>((in_shape_d2 + (dim_len - 1)) / dim_len)
    cdef double max_val = np.max(input_np)
    cdef double min_val = np.min(input_np)
    cdef double quantize_level = (max_val - min_val) / (pow(2.0, quantize_bits) - 1.0)
    cdef np.ndarray[F64_t, ndim=2] src_ = (input_np - min_val) / quantize_level
    cdef int64_t[:, :] src_round = (src_ + 0.5).astype(I64)
    cdef uint64_t[:, :, :] src_ui = np.zeros([in_shape_d1, quantize_bits, bitset], dtype=UI64)

    #  quantize input
    for batch in range(in_shape_d1):
        for dim in range(in_shape_d2):
            dim_ui = <int>(dim / dim_len)
            for exp in range(quantize_bits):
                src_ui[batch, exp, dim_ui] = ((src_ui[batch, exp, dim_ui]) << <uint64_t>1) \
                                            + ((src_round[batch, dim] >> <int64_t>exp) & <int64_t>1)

    cdef double[:] restore_coeff = quantize_level * np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)
    cdef double[:, :] app_bitc = np.zeros([in_shape_d1, quantize_bits])

    for batch in range(in_shape_d1):
        for exp in range(quantize_bits):
            for dim_ui in range(bitset):
                app_bitc[batch, exp] += <double>_mm_popcnt_u64(src_ui[batch, exp, dim_ui])

    cdef int d1, k
    cdef double Mx = 0.0, accum_dist = 0.0
    cdef uint64_t Mx_and
    cdef double[:, :] ret = np.zeros([in_shape_d1, w_shape_d1])

    #  popcnt
    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for batch in range(in_shape_d1):
            for d1 in prange(w_shape_d1, schedule='dynamic'):
                for k in range(basis):
                    Mx = 0.0
                    for exp in range(quantize_bits):
                        accum_dist = 0.0
                        for dim_ui in range(bitset):
                            Mx_and = src_ui[batch, exp, dim_ui] & M_ui[d1, k, dim_ui]
                            accum_dist += <double>_mm_popcnt_u64(Mx_and) # mosiikasite <double> -> <int>
                        Mx += (2.0 * accum_dist - app_bitc[batch, exp]) * restore_coeff[exp]
                    ret[batch, d1] += c[d1, k] * Mx
                # ret[batch, d1] += offset[d1] * min_val

    return ret


@cython.boundscheck(False)  # overflow/underflow check off
@cython.wraparound(False)  #  negative index check off
@cython.nonecheck(False)  # none check off
@cython.cdivision(True)  # cdivision on
cdef double[:, :] linear_binary_product_2axis(np.ndarray[F64_t, ndim=2] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    
    cdef int dim, dim1, dim2, exp, exp2, dim_bit, k, wdim1
    cdef double Mx = 0.0, accum_dist = 0.0, qlevel_minmax = 0.0, quantize_level = 0.0
    cdef uint64_t Mx_and = 0
    cdef int64_t src_round = 0
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef long bitset = <long>((in_shape_d2 + (dim_len - 1)) / dim_len)
    cdef double qlevel_base = pow(2.0, quantize_bits) - 1.0


    cdef int[:] dim_ui = np.zeros(in_shape_d2, dtype=I32)
    cdef double[:, :] ret = np.zeros([in_shape_d1, w_shape_d1], dtype=F64)
    cdef double[:, :] restore_coeff = np.zeros([in_shape_d1, quantize_bits], dtype=F64)
    cdef double[:, :] input_np_ = input_np
    cdef double[:] max_val = np.max(input_np, axis=1)
    cdef double[:] min_val = np.min(input_np, axis=1)
    cdef uint64_t[:, :, :] src_ui = np.zeros([in_shape_d1, quantize_bits, bitset], dtype=UI64)
    cdef double[:, :] app_bitc = np.zeros([in_shape_d1, quantize_bits], dtype=F64)
    cdef double[:] powers = np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)
    for dim in range(in_shape_d2):
        dim_ui[dim] = <int>(dim / dim_len)

    #with nogil:
    with nogil, parallel(num_threads=1):
        for dim1 in range(in_shape_d1):
            quantize_level = (max_val[dim1] - min_val[dim1]) / qlevel_base
            for dim2 in range(in_shape_d2):
                for exp in range(quantize_bits):
                    src_round = <int64_t>(((input_np_[dim1, dim2] - min_val[dim1]) / quantize_level) + 0.5)
                    src_ui[dim1, exp, dim_ui[dim2]] = ((src_ui[dim1, exp, dim_ui[dim2]]) << <uint64_t>1) \
                                                      +((src_round >> <int64_t>exp) & <int64_t>1)
                    restore_coeff[dim1, exp] = quantize_level * powers[exp]
            for exp in range(quantize_bits):
                for dim_bit in range(bitset):
                    app_bitc[dim1, exp] += <double>_mm_popcnt_u64(src_ui[dim1, exp, dim_bit])


    #  popcnt
    with nogil, parallel(num_threads=1):
    #with nogil, parallel(num_threads=num_threads):
            for dim1 in range(in_shape_d1):
                for wdim1 in prange(w_shape_d1, schedule='dynamic'):
                    for k in range(basis):
                        Mx = 0.0
                        for exp in range(quantize_bits):
                            accum_dist = 0.0
                            for dim_bit in range(bitset):
                                Mx_and = src_ui[dim1, exp, dim_bit] & M_ui[wdim1, k, dim_bit]
                                accum_dist += <double>_mm_popcnt_u64(Mx_and) # mosiikasite <double> -> <int>
                            Mx += (2.0 * accum_dist - app_bitc[dim1, exp]) * restore_coeff[dim1, exp]
                        ret[dim1, wdim1] += c[wdim1, k] * Mx
                    ret[dim1, wdim1] += offset[wdim1] * min_val[dim1]
    return ret

@cython.boundscheck(False)  # overflow/underflow check off
@cython.wraparound(False)  #  negative index check off
@cython.nonecheck(False)  # none check off
@cython.cdivision(True)  # cdivision on
cdef double[:, :, :, :] linear_binary_product_4float(np.ndarray[F64_t, ndim=4] input_np, np.ndarray[F64_t, ndim=2] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):

    cdef int i, j, k, m, n
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2]
    cdef int in_shape_d4 = input_np.shape[3]
    cdef int w_shape_d1 = weight_np.shape[0]
    cdef int w_shape_d2 = weight_np.shape[1]
    
    cdef double[:, :, :, :] ret = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3, w_shape_d1],dtype=F64)

    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for i in range(in_shape_d1):
            for j in range(in_shape_d2):
                for k in range(in_shape_d3):
                    for m in range(w_shape_d1):
                        ret[i, j, k, m] = 0.0
                        for n in range(in_shape_d4):
                            ret[i,j, k, m] += input_np[i,j,k,n] * weight_np[m,n]

    return ret

@cython.boundscheck(False)  # overflow/underflow check off
@cython.wraparound(False)  #  negative index check off
@cython.nonecheck(False)  # none check off
@cython.cdivision(True)  # cdivision on
cdef double[:, :, :, :] linear_binary_product_4axis(np.ndarray[F64_t, ndim=4] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    
    cdef int dim, dim1, dim2, dim3, dim4, exp, dim_bit, k, wdim1
    cdef double Mx = 0.0, accum_dist = 0.0, qlevel_minmax = 0.0, quantize_level = 0.0
    cdef uint64_t Mx_and = 0
    cdef int64_t src_round = 0
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2]
    cdef int in_shape_d4 = input_np.shape[3]
    cdef long bitset = <long>((in_shape_d4 + (dim_len - 1)) / dim_len)
    cdef double qlevel_base = pow(2.0, quantize_bits) - 1.0

    cdef int[:] dim_ui = np.zeros(in_shape_d4, dtype=I32)
    cdef double[:, :, :, :] ret = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3, w_shape_d1], dtype=F64)
    cdef double[:, :, :, :] restore_coeff= np.zeros([in_shape_d1, in_shape_d2, in_shape_d3, quantize_bits], dtype=F64)
    cdef double[:, :, :, :] input_np_ = input_np
    cdef double[:, :, :] max_val = np.max(input_np, axis=3)
    cdef double[:, :, :] min_val = np.min(input_np, axis=3)
    cdef uint64_t[:, :, :, :, :] src_ui = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3, quantize_bits, bitset], dtype=UI64)
    cdef double[:, :, :, :] app_bitc = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3, quantize_bits], dtype=F64)
    cdef double[:] powers = np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)
    for dim in range(in_shape_d4):
        dim_ui[dim] = <int>(dim / dim_len)

    #with nogil:
    with nogil, parallel(num_threads=1):
        for dim1 in range(in_shape_d1):
            for dim2 in range(in_shape_d2):
                for dim3 in range(in_shape_d3):
                    quantize_level = (max_val[dim1, dim2, dim3] - min_val[dim1, dim2, dim3]) / qlevel_base
                    for dim4 in range(in_shape_d4):
                        for exp in range(quantize_bits):
                            src_round = <int64_t>(((input_np_[dim1, dim2, dim3, dim4] - min_val[dim1, dim2, dim3]) / quantize_level) + 0.5)
                            src_ui[dim1, dim2, dim3, exp, dim_ui[dim4]] = ((src_ui[dim1, dim2, dim3, exp, dim_ui[dim4]]) << <uint64_t>1) \
                                                                          +((src_round >> <int64_t>exp) & <int64_t>1)
                            restore_coeff[dim1, dim2, dim3, exp] = quantize_level * powers[exp]
                    for exp in range(quantize_bits):
                        for dim_bit in range(bitset):
                            app_bitc[dim1, dim2, dim3, exp] += <double>_mm_popcnt_u64(src_ui[dim1, dim2, dim3, exp, dim_bit])


    #  popcnt
    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for dim1 in range(in_shape_d1):
            for dim2 in range(in_shape_d2):
                for dim3 in range(in_shape_d3):
                    for wdim1 in prange(w_shape_d1, schedule='dynamic'):
                        for k in range(basis):
                            Mx = 0.0
                            for exp in range(quantize_bits):
                                accum_dist = 0.0
                                for dim_bit in range(bitset):
                                    Mx_and = src_ui[dim1, dim2, dim3, exp, dim_bit] & M_ui[wdim1, k, dim_bit]
                                    accum_dist += <double>_mm_popcnt_u64(Mx_and) # mosiikasite <double> -> <int>
                                Mx += (2.0 * accum_dist - app_bitc[dim1, dim2, dim3, exp]) * restore_coeff[dim1, dim2, dim3, exp]
                            ret[dim1, dim2, dim3, wdim1] += c[wdim1, k] * Mx
                        ret[dim1, dim2, dim3, wdim1] += offset[wdim1] * min_val[dim1, dim2, dim3]
    return ret

@cython.boundscheck(False)  # overflow/underflow check off
@cython.wraparound(False)  #  negative index check off
@cython.nonecheck(False)  # none check off
@cython.cdivision(True)  # cdivision on
cdef double[:, :, :] linear_binary_product_3float(np.ndarray[F64_t, ndim=3] input_np, np.ndarray[F64_t, ndim=2] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    
    cdef int i, j, m, n
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2]
    cdef int w_shape_d1 = weight_np.shape[0]
    cdef int w_shape_d2 = weight_np.shape[1]
    
    cdef double[:, :, :] ret = np.zeros([in_shape_d1, in_shape_d2, w_shape_d1],dtype=F64)

    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for i in range(in_shape_d1):
            for j in range(in_shape_d2):
                    for m in range(w_shape_d1):
                        ret[i, j, m] = 0.0
                        for n in range(in_shape_d3):
                            ret[i,j, m] += input_np[i,j,n] * weight_np[m,n]

    return ret

@cython.boundscheck(False)  # overflow/underflow check off
@cython.wraparound(False)  #  negative index check off
@cython.nonecheck(False)  # none check off
@cython.cdivision(True)  # cdivision on
cdef double[:, :, :] linear_binary_product_3axis(np.ndarray[F64_t, ndim=3] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """

    cdef int dim, dim1, dim2, dim3, exp, exp2, dim_bit, k, wdim1
    cdef double Mx = 0.0, accum_dist = 0.0, qlevel_minmax = 0.0, quantize_level = 0.0
    cdef uint64_t Mx_and = 0
    cdef int64_t src_round = 0
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2]
    cdef long bitset = <long>((in_shape_d3 + (dim_len - 1)) / dim_len)
    cdef double qlevel_base = pow(2.0, quantize_bits) - 1.0

    cdef int[:] dim_ui = np.zeros(in_shape_d3, dtype=I32)
    cdef double[:, :, :] ret = np.zeros([in_shape_d1, in_shape_d2, w_shape_d1], dtype=F64)
    cdef double[:, :, :] restore_coeff = np.zeros([in_shape_d1, in_shape_d2, quantize_bits], dtype=F64)
    cdef double[:, :, :] input_np_ = input_np
    cdef double[:, :] max_val = np.max(input_np, axis=2)
    cdef double[:, :] min_val = np.min(input_np, axis=2)
    cdef uint64_t[:, :, :, :] src_ui = np.zeros([in_shape_d1, in_shape_d2, quantize_bits, bitset], dtype=UI64)
    cdef double[:, :, :] app_bitc = np.zeros([in_shape_d1, in_shape_d2, quantize_bits], dtype=F64)
    cdef double[:] powers = np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)
    for dim in range(in_shape_d3):
        dim_ui[dim] = <int>(dim / dim_len)

    #with nogil:
    with nogil, parallel(num_threads=1):
        for dim1 in prange(in_shape_d1, schedule='dynamic'):
            for dim2 in range(in_shape_d2):
                quantize_level = (max_val[dim1, dim2] - min_val[dim1, dim2]) / qlevel_base
                for dim3 in range(in_shape_d3):
                    for exp in range(quantize_bits):
                        src_round = <int64_t>(((input_np_[dim1, dim2, dim3] - min_val[dim1, dim2]) / quantize_level) + 0.5)
                        src_ui[dim1, dim2, exp, dim_ui[dim3]] = ((src_ui[dim1, dim2, exp, dim_ui[dim3]]) << <uint64_t>1) \
                                                                +((src_round >> <int64_t>exp) & <int64_t>1)
                        restore_coeff[dim1, dim2, exp] = quantize_level * powers[exp]
                for exp in range(quantize_bits):
                    for dim_bit in range(bitset):
                        app_bitc[dim1, dim2, exp] += <double>_mm_popcnt_u64(src_ui[dim1, dim2, exp, dim_bit])


    #  popcnt
    with nogil, parallel(num_threads=1):
    #with nogil, parallel(num_threads=num_threads):
            for dim1 in range(in_shape_d1):
                for dim2 in range(in_shape_d2):
                    for wdim1 in prange(w_shape_d1, schedule='dynamic'):
                        for k in range(basis):
                            Mx = 0.0
                            for exp in range(quantize_bits):
                                accum_dist = 0.0
                                for dim_bit in range(bitset):
                                    Mx_and = src_ui[dim1, dim2, exp, dim_bit] & M_ui[wdim1, k, dim_bit]
                                    accum_dist += <double>_mm_popcnt_u64(Mx_and) # mosiikasite <double> -> <int>
                                Mx += (2.0 * accum_dist - app_bitc[dim1, dim2, exp]) * restore_coeff[dim1, dim2, exp]
                            ret[dim1, dim2, wdim1] += c[wdim1, k] * Mx
                        ret[dim1, dim2, wdim1] += offset[wdim1] * min_val[dim1, dim2]
    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#  RNN/LSTM binary product
cdef double[:, :] linear_binary_product_RNN_func(np.ndarray[F64_t, ndim=2] input_np, uint64_t[:, :, :] src_ui, \
                    double[:, :] app_bitc, double[:] restore_coeff, int w_shape_d1, uint64_t[:, :, :] M_ui, double[:, :] c, \
                    int quantize_bits, int basis, int dim_len, int num_threads):
    r"""
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    
    cdef int batch, dim_ui, exp
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef long bitset = <long>((in_shape_d2 + (dim_len - 1)) / dim_len)

    cdef int d1, k
    cdef double Mx = 0.0, accum_dist = 0.0
    cdef uint64_t Mx_and
    cdef double[:, :] ret = np.zeros([in_shape_d1, w_shape_d1])

    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for batch in range(in_shape_d1):
            for d1 in prange(w_shape_d1, schedule='dynamic'):
                for k in range(basis):
                    Mx = 0.0
                    for exp in range(quantize_bits):
                        accum_dist = 0.0
                        for dim_ui in range(bitset):
                            Mx_and =  M_ui[d1, k, dim_ui] & src_ui[batch, exp, dim_ui]
                            accum_dist += <double>_mm_popcnt_u64(Mx_and)
                        Mx += (2.0 * accum_dist - app_bitc[batch, exp]) * restore_coeff[exp]
                    ret[batch, d1] += c[d1, k] * Mx

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#  QsL for linear
cdef tuple Quantization_Sub_Layer_func(np.ndarray[F64_t, ndim=2] input_np, int quantize_bits, int dim_len, int num_threads):
    cdef int batch, dim, dim_ui, exp
    cdef int in_shape_d1 = input_np.shape[0]
    cdef int in_shape_d2 = input_np.shape[1]
    cdef long bitset = <long>((in_shape_d2 + (dim_len - 1)) / dim_len)
    cdef double max_val = np.max(input_np)
    cdef double min_val = np.min(input_np)
    cdef double quantize_level = (max_val - min_val) / (pow(2.0, quantize_bits) - 1.0)
    cdef np.ndarray[F64_t, ndim=2] src_ = (input_np - min_val) / quantize_level
    cdef int64_t[:, :] src_round = (src_ + 0.5).astype(I64)
    cdef uint64_t[:, :, :] src_ui = np.zeros([in_shape_d1, quantize_bits, bitset], dtype=UI64)

    for batch in range(in_shape_d1):
        for dim in range(in_shape_d2):
            dim_ui = <int>(dim / dim_len)
            for exp in range(quantize_bits):
                src_ui[batch, exp, dim_ui] = ((src_ui[batch, exp, dim_ui]) << <uint64_t>1) \
                                            + ((src_round[batch, dim] >> <int64_t>exp) & <int64_t>1)

    cdef double[:] restore_coeff = quantize_level * np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)
    cdef double[:, :] app_bitc = np.zeros([in_shape_d1, quantize_bits])

    for batch in range(in_shape_d1):
        for exp in range(quantize_bits):
            for dim_ui in range(bitset):
                app_bitc[batch, exp] += <double>_mm_popcnt_u64(src_ui[batch, exp, dim_ui])
    
    return src_ui, app_bitc, restore_coeff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#  not support conv2d offset # target
cdef double[:, :, :, :] conv2d_float_product(np.ndarray[F64_t, ndim=4] input_np, np.ndarray[F64_t, ndim=4] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    
    cdef int pad_h, pad_w, batch, batch2, och, ich, height, width, kheight, kwidth
    cdef double tmp
    cdef int in_shape_d1 = input_np.shape[0], in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2], in_shape_d4 = input_np.shape[3]
    cdef int w_shape_d1 = weight_np.shape[0], w_shape_d2 = weight_np.shape[1]
    cdef int w_shape_d3 = weight_np.shape[2], w_shape_d4 = weight_np.shape[3]

    cdef int out_shape_d1 = in_shape_d1, out_shape_d2 = w_shape_d1
    cdef int out_shape_d3 = <int>((in_shape_d3 + 2 * padding_h - dilation_h * (w_shape_d3 - 1) - 1) / stride_h + 1)
    cdef int out_shape_d4 = <int>((in_shape_d4 + 2 * padding_w - dilation_w * (w_shape_d4 - 1) - 1) / stride_w + 1)
    cdef np.ndarray[F64_t, ndim=4] ret = np.zeros([out_shape_d1, out_shape_d2, out_shape_d3, out_shape_d4], dtype=F64)

    # padding
    cdef np.ndarray[F64_t, ndim=4] pad_src = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3+(2*padding_h),\
                                                       in_shape_d4+(2*padding_w)], dtype=F64)
    if  1 <= padding_h or 1 <= padding_w:
        for batch in range(in_shape_d1):
            for ch in range(in_shape_d2):
                for height in range(in_shape_d3):
                    pad_h = padding_h + height
                    for width in range(in_shape_d4):
                        pad_w = padding_w + width
                        pad_src[batch, ch, pad_h, pad_w] = input_np[batch, ch, height, width]
    else:
        pad_src = input_np

    with nogil, parallel(num_threads=1):
    #with nogil, parallel(num_threads=num_threads):
        for batch2 in range(out_shape_d1):
            for och in range(out_shape_d2):
                for height in range(out_shape_d3):
                    for width in range(out_shape_d4):
                        for ich in range(in_shape_d2):
                            for kheight in range(w_shape_d3):
                                for kwidth in range(w_shape_d4):
                                    ret[batch2, och, height, width] += pad_src[batch2, ich, height * stride_h + kheight * dilation_h, width * stride_w + kwidth * dilation_w] * weight_np[och, ich, kheight, kwidth]

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#  not support conv2d offset # target
cdef double[:, :, :, :] conv2d_binary_product(np.ndarray[F64_t, ndim=4] input_np, int[:] w_shape, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    
    cdef int batch, dim, dim_ui, exp, d2, d3, s1, s2, s3, w3, w4, pad_h, pad_w
    cdef int s2_begin, s3_begin
    cdef int in_shape_d1 = input_np.shape[0], in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2], in_shape_d4 = input_np.shape[3]
    cdef int kh = w_shape[2], kw = w_shape[3]
    cdef int out_shape_d1 = in_shape_d1, out_shape_d2 = w_shape[0]
    cdef int out_shape_d3 = <int>((in_shape_d3 - kh + (2 * padding_h)) / stride_h + 1)
    cdef int out_shape_d4 = <int>((in_shape_d4 - kw + (2 * padding_w)) / stride_w + 1)
    cdef int in_size = in_shape_d2 * kh * kw
    cdef int bitset = (in_size + (dim_len - 1)) // dim_len
    
    # padding
    cdef np.ndarray[F64_t, ndim=4] pad_src = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3+(2*padding_h),\
                                                       in_shape_d4+(2*padding_w)], dtype=F64)
    if  1 <= padding_h or 1 <= padding_w:
        for batch in range(in_shape_d1):
            for ch in range(in_shape_d2):
                for height in range(in_shape_d3):
                    pad_h = padding_h + height
                    for width in range(in_shape_d4):
                        pad_w = padding_w + width
                        pad_src[batch, ch, pad_h, pad_w] = input_np[batch, ch, height, width]
    else:
        pad_src = input_np

    cdef double max_val = np.max(pad_src)
    cdef double min_val = np.min(pad_src)
    cdef double quantize_level = (max_val - min_val) / (pow(2.0, quantize_bits) - 1.0)
    cdef np.ndarray[F64_t, ndim=4] src_ = (pad_src - min_val) / quantize_level
    cdef int64_t[:, :, :, :] src_round = (src_ + 0.5).astype(I64)
    cdef double[:, :, :, :] x_norm = np.zeros([out_shape_d1, out_shape_d3, out_shape_d4, quantize_bits], dtype=F64)
    cdef uint64_t[:, :, :, :, :] src_ui = np.zeros([out_shape_d1, out_shape_d3, out_shape_d4 ,quantize_bits, bitset], dtype=UI64)
    
    #with nogil:
    with nogil, parallel(num_threads=1):
        for batch in range(out_shape_d1):
            for d2 in range(out_shape_d3):
                s2_begin = d2 * stride_h
                for d3 in range(out_shape_d4):
                    s3_begin = d3 * stride_w
                    dim = 0
                    for s1 in range(in_shape_d2):
                        for w3 in range(kh):
                            s2 = s2_begin + w3
                            for w4 in range(kw):
                                s3 = s3_begin + w4
                                dim_ui = <int>(dim / dim_len)
                                dim = dim + 1
                                for exp in range(quantize_bits):
                                    src_ui[batch, d2, d3, exp, dim_ui] = ((src_ui[batch, d2, d3, exp, dim_ui]) << <uint64_t>1) \
                                                                + ((src_round[batch, s1, s2, s3] >> <int64_t>exp) & <int64_t>1)
                    for exp in range(quantize_bits):
                        x_norm[batch, d2, d3, exp] = 0.0
                        for dim_ui in range(bitset):
                            x_norm[batch, d2, d3, exp] += <double>_mm_popcnt_u64(src_ui[batch, d2, d3, exp, dim_ui])

    cdef double[:] restore_coeff = quantize_level * np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)

    cdef int d1, k
    cdef double Mx = 0.0, accum_dist = 0.0
    cdef uint64_t Mx_and
    cdef double[:, :, :, :] ret = np.zeros([out_shape_d1, out_shape_d2, out_shape_d3, out_shape_d4], dtype=F64)

    #  popcnt
    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for batch in range(in_shape_d1):
            for d1 in prange(out_shape_d2, schedule='dynamic'):
                for d2 in range(out_shape_d3):
                    for d3 in range(out_shape_d4):
                        for k in range(basis):
                            Mx = 0.0
                            for exp in range(quantize_bits):
                                accum_dist = 0.0
                                for dim_ui in range(bitset):
                                    Mx_and = src_ui[batch, d2, d3, exp, dim_ui] & M_ui[d1, k, dim_ui]
                                    accum_dist += <double>_mm_popcnt_u64(Mx_and)
                                Mx += (2.0 * accum_dist - x_norm[batch, d2, d3, exp]) * restore_coeff[exp]
                            ret[batch, d1, d2, d3] += c[d1, k] * Mx

    return ret



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
#  support conv2d offset # target
cdef double[:, :, :, :] conv2d_binary_product_offset(np.ndarray[F64_t, ndim=4] input_np, int[:] w_shape, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    
    cdef int batch, dim, dim_ui, exp, d2, d3, s1, s2, s3, w3, w4, pad_h, pad_w
    cdef int s2_begin, s3_begin
    cdef int in_shape_d1 = input_np.shape[0], in_shape_d2 = input_np.shape[1]
    cdef int in_shape_d3 = input_np.shape[2], in_shape_d4 = input_np.shape[3]
    cdef int kh = w_shape[2], kw = w_shape[3]
    cdef int out_shape_d1 = in_shape_d1, out_shape_d2 = w_shape[0]
    cdef int out_shape_d3 = <int>((in_shape_d3 - kh + (2 * padding_h)) / stride_h + 1)
    cdef int out_shape_d4 = <int>((in_shape_d4 - kw + (2 * padding_w)) / stride_w + 1)
    cdef int in_size = in_shape_d2 * kh * kw
    cdef int bitset = (in_size + (dim_len - 1)) // dim_len

    #  padding
    cdef np.ndarray[F64_t, ndim=4] pad_src = np.zeros([in_shape_d1, in_shape_d2, in_shape_d3+(2*padding_h),\
                                                       in_shape_d4+(2*padding_w)], dtype=F64)
    if  1 <= padding_h or 1 <= padding_w:
        for batch in range(in_shape_d1):
            for ch in range(in_shape_d2):
                for height in range(in_shape_d3):
                    pad_h = padding_h + height
                    for width in range(in_shape_d4):
                        pad_w = padding_w + width
                        pad_src[batch, ch, pad_h, pad_w] = input_np[batch, ch, height, width]
    else:
        pad_src = input_np

    cdef double max_val = np.max(pad_src)
    cdef double min_val = np.min(pad_src)
    cdef double quantize_level = (max_val - min_val) / (pow(2.0, quantize_bits) - 1.0)
    cdef np.ndarray[F64_t, ndim=4] src_ = (pad_src - min_val) / quantize_level
    cdef int64_t[:, :, :, :] src_round = (src_ + 0.5).astype(I64)
    cdef double[:, :, :, :] x_norm = np.zeros([out_shape_d1, out_shape_d3, out_shape_d4, quantize_bits], dtype=F64)
    cdef uint64_t[:, :, :, :, :] src_ui = np.zeros([out_shape_d1, out_shape_d3, out_shape_d4 ,quantize_bits, bitset], dtype=UI64)
    
    #with nogil:
    with nogil, parallel(num_threads=1):
        for batch in range(out_shape_d1):
            for d2 in range(out_shape_d3):
                s2_begin = d2 * stride_h
                for d3 in range(out_shape_d4):
                    s3_begin = d3 * stride_w
                    dim = 0
                    for s1 in range(in_shape_d2):
                        for w3 in range(kh):
                            s2 = s2_begin + w3
                            for w4 in range(kw):
                                s3 = s3_begin + w4
                                dim_ui = <int>(dim / dim_len)
                                dim = dim + 1
                                for exp in range(quantize_bits):
                                    src_ui[batch, d2, d3, exp, dim_ui] = ((src_ui[batch, d2, d3, exp, dim_ui]) << <uint64_t>1) \
                                                                + ((src_round[batch, s1, s2, s3] >> <int64_t>exp) & <int64_t>1)
                    for exp in range(quantize_bits):
                        x_norm[batch, d2, d3, exp] = 0.0
                        for dim_ui in range(bitset):
                            x_norm[batch, d2, d3, exp] += <double>_mm_popcnt_u64(src_ui[batch, d2, d3, exp, dim_ui])

    cdef double[:] restore_coeff = quantize_level * np.power(2.0, np.arange(quantize_bits, dtype=F64), dtype=F64)

    cdef int d1, k
    cdef double Mx = 0.0, accum_dist = 0.0, offset_
    cdef uint64_t Mx_and
    cdef double[:, :, :, :] ret = np.zeros([out_shape_d1, out_shape_d2, out_shape_d3, out_shape_d4], dtype=F64)

    #  popcnt
    with nogil, parallel(num_threads=1):
#    with nogil, parallel(num_threads=num_threads):
        for batch in range(out_shape_d1):
            for d1 in prange(out_shape_d2, schedule='dynamic'):
                offset_ = offset[d1] * min_val
                for d2 in range(out_shape_d3):
                    for d3 in range(out_shape_d4):
                        for k in range(basis):
                            Mx = 0.0
                            for exp in range(quantize_bits):
                                accum_dist = 0.0
                                for dim_ui in range(bitset):
                                    Mx_and = M_ui[d1, k, dim_ui] & src_ui[batch, d2, d3, exp, dim_ui]
                                    accum_dist += <double>_mm_popcnt_u64(Mx_and)
                                Mx += (2.0 * accum_dist - x_norm[batch, d2, d3, exp]) * restore_coeff[exp]
                            ret[batch, d1, d2, d3] += c[d1, k] * Mx
                        ret[batch, d1, d2, d3] = ret[batch, d1, d2, d3] + offset_

    return ret


cpdef double[:, :] linear_binary_popcnt(np.ndarray[F64_t, ndim=2] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product(input_np, w_shape_d1, M_ui, c, offset, quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :, :, :] linear_binary_popcnt_4axis(np.ndarray[F64_t, ndim=4] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product_4axis(input_np, w_shape_d1, M_ui, c, offset, quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :, :, :] linear_binary_popcnt_4float(np.ndarray[F64_t, ndim=4] input_np, np.ndarray[F64_t, ndim=2] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product_4float(input_np, weight_np, M_ui, c, offset, quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :, :] linear_binary_popcnt_3axis(np.ndarray[F64_t, ndim=3] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product_3axis(input_np, w_shape_d1, M_ui, c, offset, quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :, :] linear_binary_popcnt_3float(np.ndarray[F64_t, ndim=3] input_np, np.ndarray[F64_t, ndim=2] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product_3float(input_np, weight_np, M_ui, c, offset, quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :] linear_binary_popcnt_2axis(np.ndarray[F64_t, ndim=2] input_np, int w_shape_d1, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product_2axis(input_np, w_shape_d1, M_ui, c, offset, quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :] linear_binary_popcnt_RNN(np.ndarray[F64_t, ndim=2] input_np, uint64_t[:, :, :] src_ui, \
                    double[:, :] app_bitc, double[:] restore_coeff, int w_shape_d1, uint64_t[:, :, :] M_ui, double[:, :] c, \
                    int quantize_bits, int basis, int dim_len, int num_threads):
    return linear_binary_product_RNN_func(input_np, src_ui, app_bitc, restore_coeff, w_shape_d1, M_ui, c, quantize_bits, basis, dim_len, num_threads)

cpdef tuple Quantization_Sub_Layer(np.ndarray[F64_t, ndim=2] input_np, int quantize_bits, int dim_len, int num_threads):
    return Quantization_Sub_Layer_func(input_np, quantize_bits, dim_len, num_threads)

cpdef double[:, :, :, :] conv2d_float_popcnt(np.ndarray[F64_t, ndim=4] input_np, np.ndarray[F64_t, ndim=4] weight_np, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    return conv2d_float_product(input_np, weight_np, M_ui, c, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,\
    quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :, :, :] conv2d_binary_popcnt(np.ndarray[F64_t, ndim=4] input_np, int[:] w_shape, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    return conv2d_binary_product(input_np, w_shape, M_ui, c, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,\
    quantize_bits, basis, dim_len, num_threads)

cpdef double[:, :, :, :] conv2d_binary_popcnt_offset(np.ndarray[F64_t, ndim=4] input_np, int[:] w_shape, uint64_t[:, :, :] M_ui,\
                        double[:, :] c, double[:] offset, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
                        int groups, int quantize_bits, int basis, int dim_len, int num_threads):
    return conv2d_binary_product_offset(input_np, w_shape, M_ui, c, offset, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,\
    quantize_bits, basis, dim_len, num_threads)

#cpdef double[:, :, :, :] conv2d_float32(np.ndarray[F64_t, ndim=4] input_np, np.ndarray[F64_t, ndim=4] weight_np, uint64_t[:, :, :] M_ui,\
   #                     double[:, :] c, int stride_h, int stride_w, int padding_h, int padding_w, int dilation_h, int dilation_w,\
  #                      int groups, int quantize_bits, int basis, int dim_len, int num_threads):
  #  return conv2d_before_float_product(input_np, weight_np, M_ui, c, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups,\
  #  quantize_bits, basis, dim_len, num_threads)