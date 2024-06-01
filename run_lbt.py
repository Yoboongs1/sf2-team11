import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.run_dct import dctbpp, dct_matrix
from cued_sf2_lab.quality_measure import uiqi, psnr, rms, calc_ssim

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

def lbt(N, X):
    C = dct_matrix(N)
    Pf, Pr = pot_ii(N)
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T
    Z = colxfm(colxfm(Y.T, C.T).T, C.T)
    Zp = Z.copy()  #copy the non-transformed edges directly from Z
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Zp

def lbt_quant(N, X, s, step_size):
    C = dct_matrix(N)
    Pf, Pr = pot_ii(N,s)
    t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
    Xp = X.copy()  # copy the non-transformed edges directly from X
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    Y = colxfm(colxfm(Xp, C).T, C).T
    Yq = quantise(Y,step_size)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    Zp = Z.copy()  #copy the non-transformed edges directly from Z
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)
    return Xp,Zp

def optimal_step_size(N, X, s):
    Xq = quantise(X,17)
    error = np.std(X-Xq)
    min_diff_error = 100
    min_diff_step_size = 0
    step_size = np.arange(5,30.1,0.1)
    for step in step_size:
        Xp, Zp = lbt_quant(N, X, s, step)
        new_error = np.std(Zp-X)
        if abs(new_error - error) < min_diff_error:
            min_diff_error = abs(new_error - error)
            min_diff_step_size = step
    return min_diff_step_size

def optimal_s_bits_compression_ratio(N, X):
    C = dct_matrix(N)
    optimal_step_list = []
    compression_ratio_list = []
    bits_list = []
    Xq = quantise(X,17)
    error = np.std(X-Xq)
    bits_quant = bpp(Xq)*(len(Xq)**2)
    Y = colxfm(colxfm(X, C).T, C).T
    s_range = np.arange(1,2.05,0.05)

    for s in s_range:
        optimal_step = optimal_step_size(N, X, s)
        optimal_step_list.append(optimal_step)
        Pf, Pr = pot_ii(N, s)
        t = np.s_[N//2:-N//2]
        Xp = X.copy()
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
        Y = colxfm(colxfm(Xp, C).T, C).T
        Yq = quantise(Y,optimal_step)
        Yr = regroup(Yq,N)/N
        bits_dct = dctbpp(Yr,16)
        bits_list.append(bits_dct)
        compression_ratio_list.append(bits_quant/bits_dct)

    max_compression_ratio = max(compression_ratio_list)
    optimal_s = s_range[np.argmax(compression_ratio_list)]
    optimal_step = optimal_step_list[np.argmax(compression_ratio_list)]
    optimal_bits = bits_list[np.argmax(compression_ratio_list)]
    return max_compression_ratio, optimal_s, optimal_bits, optimal_step

def plot_optimal_image(X,N, s, optimal_step):
    Xp, Zp = lbt_quant(N, X, s, optimal_step)
    fig,ax = plt.subplots(1,2)
    plot_image(X, ax=ax[0])
    ax[0].set(title='Original')
    plot_image(Zp, ax=ax[1])
    ax[1].set(title='LBT Compressed')
    plt.show()
    return Zp

def test_image(X,N):
    X = X - 128.0
    comp,s,bits, optimal_step = optimal_s_bits_compression_ratio(N,X)
    print(f"Optimal s value: {s}")
    print(f"Number of bits required: {bits}")
    print(f"Compression ratio: {comp}")
    Z = plot_optimal_image(X,N,s,optimal_step)
    print(f"RMS: {rms(X,Z)}")
    print(f"UIQI: {uiqi(X,Z)}")
    print(f"PSNR: {psnr(X,Z)}")
    print(f"SSIM: {calc_ssim(X,Z)}")
    print("--------------")

""" test_image(lighthouse,8)
test_image(bridge,8)
test_image(flamingo,8) """