import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Load original image
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Add noise
noise = np.random.normal(0, 20, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

# Median + TV denoising
median_out = median_filter(noisy, size=3)
tv_out = denoise_tv_chambolle(median_out / 255.0, weight=0.1)
tv_out = (tv_out * 255).astype(np.uint8)

# Metrics
psnr = peak_signal_noise_ratio(img, tv_out)
ssim = structural_similarity(img, tv_out)

print("PSNR:", psnr)
print("SSIM:", ssim)

