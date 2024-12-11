import numpy as np
from numba import prange
import numba as nb
import warnings
warnings.filterwarnings('ignore')

@nb.jit(nb.float64[:, :](nb.float64[:], nb.float64[:], nb.float64[:, :]), nopython=True, nogil=True, parallel=True)
def M_uplode(ws, cs, all_candi):
    best_candies = np.zeros((ws.size, cs.size), dtype=np.float64)
    all_candi_real = np.dot(all_candi, cs)
    ws_size = ws.size
    
    for i in prange(ws_size):
        best_idx = np.abs(all_candi_real - ws[i]).argmin()
        best_candies[i] = all_candi[best_idx]
        
    return best_candies

@nb.jit(nb.types.Tuple((nb.float64[:, :], nb.float64[:]))(nb.float64[:], nb.int64, nb.float64[:, :]), nopython=True, nogil=True)
def Exhaustive_decompose(w, k, all_candidates2):
    opt_M = np.zeros((w.size, k), dtype=np.float64)
    opt_c = np.zeros(k, dtype=np.float64)
    min_e = np.inf
    
    max_iter = 75
    init_iter = 65

    for n in prange(init_iter):
        M = np.random.randn(w.size, k)
        M = M / np.abs(M)
        for iters in prange(max_iter):
            c = np.dot(np.linalg.pinv(np.dot(M.T, M)), (np.dot(M.T, w)))
            M = M_uplode(w, c, all_candidates2)
            cost = np.dot((w - np.dot(M, c)).T, (w - np.dot(M, c)))
            if min_e > cost:
                min_e = cost
                opt_M = M
                opt_c = c

    return opt_M, opt_c
