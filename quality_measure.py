import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity as ssim
from cued_sf2_lab.familiarisation import load_mat_img

lighthouse, _ = load_mat_img(img='lighthouse.mat', img_info='X')

def calc_ssim(original, compressed):
    ssim_value, _ = ssim(original, compressed, full=True, data_range= compressed.max() - compressed.min())
    return ssim_value

def uiqi(original, compressed):
    original_mean = np.mean(original, dtype=np.float64)
    compressed_mean = np.mean(compressed, dtype=np.float64)
    original_var = np.var(original, dtype=np.float64)
    compressed_var = np.var(compressed, dtype=np.float64)
    covariance = np.mean((original - original_mean) * (compressed - compressed_mean), dtype=np.float64)
    
    numerator = 4 * original_mean * compressed_mean * covariance
    denominator = (original_mean**2 + compressed_mean**2) * (original_var + compressed_var)
    
    if denominator == 0:
        return 1 if numerator == 0 else 0
    else:
        return numerator / denominator

def psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2, dtype=np.float64)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def rms(original, compressed):
    return np.std(original-compressed)