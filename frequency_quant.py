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

# Load images
lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')
bridge, _ = load_mat_img(img='bridge.mat', img_info='X')
flamingo, _ = load_mat_img(img='flamingo.mat', img_info='X')

# Define DCT matrix function
def dct_matrix(N):
    return dct_ii(N)

# Define JPEG luminance quantisation table
jpeg_quant_luminance = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99] 
])

# Function to apply JPEG quantisation to 8x8 blocks
def quantise_jpeg(Y, Q):
    Yq = np.zeros_like(Y)
    for i in range(0, Y.shape[0], 8):
        for j in range(0, Y.shape[1], 8):
            Yq[i:i+8, j:j+8] = np.round(Y[i:i+8, j:j+8] / Q)
    return Yq

# Function to dequantise JPEG quantised 8x8 blocks
def dequantise_jpeg(Yq, Q):
    Y = np.zeros_like(Yq)
    for i in range(0, Yq.shape[0], 8):
        for j in range(0, Yq.shape[1], 8):
            Y[i:i+8, j:j+8] = Yq[i:i+8, j:j+8] * Q
    return Y

# Function to calculate bits per pixel (bpp) for a DCT-transformed image
def dctbpp(Yr, N):
    total_entropy = 0
    sub_image_interval = len(Yr) // N
    for i in range(N):
        for j in range(N):
            Ys = Yr[sub_image_interval * i : sub_image_interval * (i + 1), sub_image_interval * j : sub_image_interval * (j + 1)]
            entropy = bpp(Ys) * (len(Ys) ** 2)
            total_entropy += entropy
    return total_entropy

# Function to calculate compression ratio using JPEG quantisation
def calculate_compression_ratio(X, N, Q):
    C = dct_matrix(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yq = quantise_jpeg(Y, Q)
    Yr = regroup(Yq,N)/N
    bits_dct = dctbpp(Yr, N)
    Xq = quantise(X, 17)
    bits_quant = bpp(Xq) * (len(Xq) ** 2)
    compression_ratio = bits_quant / bits_dct
    return bits_dct, compression_ratio, Yq

# Function to plot compressed image using JPEG quantisation
def plot_compressed_image(X, N, Yq, Q):
    C = dct_matrix(N)
    Z = colxfm(colxfm(dequantise_jpeg(Yq, Q).T, C.T).T, C.T)
    fig, ax = plt.subplots(1, 2)
    plot_image(X, ax=ax[0])
    ax[0].set(title='Original')
    plot_image(Z, ax=ax[1])
    ax[1].set(title='DCT Compressed')
    plt.show()
    return Z

# Function to test compression on an image
def test_image(X, N, Q):
    X = X - 128.0
    bits, comp, Yq = calculate_compression_ratio(X, N, Q)
    print(f"Number of bits required: {bits}")
    print(f"Compression ratio: {comp}")
    Z = plot_compressed_image(X, N, Yq, Q)
    print(f"RMS: {rms(X, Z)}")
    print(f"UIQI: {uiqi(X, Z)}")
    print(f"PSNR: {psnr(X, Z)}")
    print(f"SSIM: {calc_ssim(X, Z)}")
    print("--------------")

# Test with different images
test_image(lighthouse, 8, jpeg_quant_luminance)
test_image(bridge, 8, jpeg_quant_luminance)
test_image(flamingo, 8, jpeg_quant_luminance)
