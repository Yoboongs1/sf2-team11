from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import regroup
import numpy as np
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.quality_measure import uiqi, psnr, rms, calc_ssim
import matplotlib.pyplot as plt

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

def dct_matrix(N):
    return dct_ii(N)    

def dctbpp(Yr, N):
    total_entropy = 0
    sub_image_interval = len(Yr)//N
    for i in range(N):
        for j in range(N):
            Ys = Yr[sub_image_interval*i:sub_image_interval*(i+1), sub_image_interval*j:sub_image_interval*(j+1)]  
            entropy = bpp(Ys)*(len(Ys)**2)
            total_entropy += entropy
    return total_entropy

def optimal_step_size(X, error, N):
    C = dct_matrix(N)
    step_size = np.arange(5, 25.01, 0.01)
    min_diff_error = 100
    min_diff_step_size = 0
    Y = colxfm(colxfm(X, C).T, C).T
    for i in step_size:
        test_Y = quantise(Y,i,rise1=i*0.8)
        test_Z = colxfm(colxfm(test_Y.T, C.T).T, C.T)
        new_error = np.std(X-test_Z)
        if abs(new_error - error) < min_diff_error:
            min_diff_error = abs(new_error - error)
            min_diff_step_size = i
    return min_diff_step_size

def calculate_compression_ratio(X, N):
    C = dct_matrix(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Xq = quantise(X, 17)
    error = np.std(X-Xq)
    optimal_step = optimal_step_size(X,error,N)
    Yq = quantise(Y,optimal_step,rise1=optimal_step*0.8)
    Yr = regroup(Yq,N)/N
    bits_dct = dctbpp(Yr,8)
    bits_quant = bpp(Xq)*(len(Xq)**2)
    compression_ratio = bits_quant/bits_dct
    return bits_dct, compression_ratio

def plot_compressed_image(X,N,optimal_step_size):
    C = dct_matrix(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yq = quantise(Y,optimal_step_size,rise1=optimal_step_size*0.8)
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)
    fig, ax = plt.subplots(1,2)
    plot_image(X, ax=ax[0])
    ax[0].set(title='Original')
    plot_image(Z, ax=ax[1])
    ax[1].set(title='DCT Compressed')
    plt.show()
    return Z

def test_image(X,N):
    X = X - 128.0
    bits, comp = calculate_compression_ratio(X,N)
    Xq = quantise(X, 17)
    error = np.std(X-Xq)
    print(f"Number of bits required: {bits}")
    print(f"Compression ratio: {comp}")
    Z = plot_compressed_image(X,N,optimal_step_size(X,error,N))
    print(f"RMS: {rms(X,Z)}")
    print(f"UIQI: {uiqi(X,Z)}")
    print(f"PSNR: {psnr(X,Z)}")
    print(f"SSIM: {calc_ssim(X,Z)}")
    print("--------------")

test_image(lighthouse,8)
test_image(bridge,8)
test_image(flamingo,8)

