import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Add Gaussian noise
noise = np.random.normal(0, 20, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

# Median Filter (Quantum-inspired median)
median_out = median_filter(noisy, size=3)

# Total Variation Denoising
tv_out = denoise_tv_chambolle(median_out / 255.0, weight=0.1)
tv_out = (tv_out * 255).astype(np.uint8)

# Display results
plt.figure(figsize=(12,3))
plt.subplot(1,4,1); plt.title("Original"); plt.imshow(img, cmap="gray")
plt.subplot(1,4,2); plt.title("Noisy"); plt.imshow(noisy, cmap="gray")
plt.subplot(1,4,3); plt.title("Median"); plt.imshow(median_out, cmap="gray")
plt.subplot(1,4,4); plt.title("Median + TV"); plt.imshow(tv_out, cmap="gray")
plt.show()

