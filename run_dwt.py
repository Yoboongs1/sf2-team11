import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
import numpy as np
from typing import Tuple
from cued_sf2_lab.laplacian_pyramid import rowdec, rowdec2, rowint, rowint2, bpp, quantise
from cued_sf2_lab.dwt import dwt, idwt
from cued_sf2_lab.quality_measure import uiqi, psnr, rms, calc_ssim

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

def multi_dwt(X,level):
    m=256
    Y=dwt(X)
    for i in range(level-1):
        m = m//2
        Y[:m,:m] = dwt(Y[:m,:m])
    return Y

def multi_idwt(Y,level):
    m = 256 // (2 ** (level - 1))
    for i in range(level - 1, 0, -1):
        Y[:m, :m] = idwt(Y[:m, :m])
        m = m * 2
    X = idwt(Y[:m, :m])
    return X

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """
    # your code here
    n = dwtstep.shape[1] - 1
    dwtent = np.zeros(dwtstep.shape)
    m = len(Y)
    Yq = Y.copy()
    bits = 0
    for i in range(n):
        m = m // 2
        Yq[m:2*m,:m] = quantise(Yq[m:2*m,:m], dwtstep[0, i])
        dwtent[0,i] = bpp(Yq[m:2*m,:m])
        bits += dwtent[0,i]*Yq[m:2*m,:m].size
        Yq[:m,m:2*m] = quantise(Yq[:m,m:2*m], dwtstep[1, i])
        dwtent[1,i] = bpp(Yq[:m,m:2*m])
        bits += dwtent[1,i]*Yq[:m,m:2*m].size
        Yq[m:2*m,m:2*m] = quantise(Yq[m:2*m,m:2*m], dwtstep[2, i])
        dwtent[2,i] = bpp(Yq[m:2*m,m:2*m])
        bits += dwtent[2,i]*Yq[m:2*m,m:2*m].size
    Yq[:m,:m] = quantise(Yq[:m,:m], dwtstep[0, n])
    dwtent[0,n] = bpp(Yq[:m,:m])
    bits += dwtent[0,n]*Yq[:m,:m].size

    return Yq, dwtent, bits

def optimal_step_size(X, error, N):
    Y = multi_dwt(X,N)
    step_size = np.arange(5, 25.01, 0.01)
    min_diff_error = 100
    min_diff_step_size = 0
    for i in step_size:
        test_Y = quantise(Y,i)
        test_Z = multi_idwt(test_Y,N)
        new_error = np.std(X-test_Z)
        if abs(new_error - error) < min_diff_error:
            min_diff_error = abs(new_error - error)
            min_diff_step_size = i
    return min_diff_step_size

step_ratio = [0.78, 0.452, 0.235, 0.12]
""" 
def comp_ratio(X, step_ratio, n):
    Y = multi_dwt(X,n)
    Xq = quantise(X,17)
    error = np.std(Xq-X)
    dwtstep = np.full((3,n+1),optimal_step_size(X, error, n))
    dwtstep = dwtstep*step_ratio
    print(dwtstep)
    Yq, dwtent, bits = quantdwt(Y,dwtstep)

comp_ratio(lighthouse, step_ratio, 4)
     """