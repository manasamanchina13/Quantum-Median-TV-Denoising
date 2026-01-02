# Quantum Median Filter for Total Variation Image Denoising

This project presents a beginner-friendly, quantum-inspired implementation
of image denoising using a Median Filter combined with Total Variation (TV).

## Project Steps
- Step 1: Total Variation (TV) denoising baseline
- Step 2: Median filtering followed by TV denoising
- Step 3: Performance evaluation using PSNR and SSIM

## Methodology
A noisy grayscale image is first processed using a median filter, which
simulates the behavior of a quantum median operation. The output is then
refined using total variation denoising to preserve edges while reducing noise.

Since the original implementation code from the reference paper is not
publicly available, this work is implemented based on the algorithmic
description provided in the literature.

## How to Run
1. Install dependencies
   pip install opencv-python numpy scipy scikit-image matplotlib

2. Run scripts step by step
   python step1_tv_baseline.py
   python step2_median_tv.py
   python step3_metrics.py

## Results
The proposed approach improves image quality by increasing PSNR and SSIM
values compared to noisy inputs, while preserving important image structures.
