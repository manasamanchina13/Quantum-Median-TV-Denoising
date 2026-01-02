import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Add Gaussian noise
noise = np.random.normal(0, 20, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

# Total Variation Denoising (baseline)
tv = denoise_tv_chambolle(noisy / 255.0, weight=0.1)
tv = (tv * 255).astype(np.uint8)

# Display results
plt.figure(figsize=(9,3))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(img, cmap="gray")
plt.subplot(1,3,2); plt.title("Noisy"); plt.imshow(noisy, cmap="gray")
plt.subplot(1,3,3); plt.title("TV Denoised"); plt.imshow(tv, cmap="gray")
plt.show()
