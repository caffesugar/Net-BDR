import os
import glob
import csv
import time
import cv2
import torch
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import core algorithm modules
from decoder import autoencodernet
from fit import fit
from helpers import (pil_to_np, np_to_var, apply_f, convert, 
                     set_random_seed, crop_to_even, add_gaussian_noise)

import warnings

warnings.filterwarnings("ignore", message=".*reflection_pad2d_backward_cuda.*")

class PhaseRetrievalEngine:
    """
    Core control engine for phase retrieval experiments.
    Manages data loading, forward physical modeling, network initialization, and metric logging.
    """
    def __init__(self, args):
        self.args = args
        
        # Specify the CUDA device based on the provided GPU ID
        if torch.cuda.is_available():
            self.device = f'cuda:{self.args.gpu_id}'
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device = 'cpu'
            self.dtype = torch.FloatTensor
        
        # Fixed internal hyperparameters
        self.lr = 0.001
        self.gamma = 1.0
        self.optim_strategy = 'bdr'
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        print(f"[System] Initialized on {self.device.upper()}. Output directory: {self.args.output_dir}")

    def get_image_paths(self):
        """Retrieve and filter valid images from the input directory."""
        if not os.path.exists(self.args.input_dir):
            raise FileNotFoundError(f"Directory not found: {self.args.input_dir}")
            
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        paths = [p for p in glob.glob(os.path.join(self.args.input_dir, '*.*')) 
                 if p.lower().endswith(valid_exts)]
        return paths

    def prepare_data(self, img_path):
        """Load image and generate Ground Truth (GT) and base tensors."""
        img_pil = Image.open(img_path).convert('L')
        img_pil = crop_to_even(img_pil)
        img_np = pil_to_np(img_pil)
        
        img_cv2_gt = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_cv2_gt = crop_to_even(img_cv2_gt)
        
        return img_np, img_cv2_gt

    def simulate_measurement(self, img_np, d, output_depth):
        """Simulate the physical measurement process (Fourier transform + noise)."""
        k_len = round(self.args.k * d)
        
        # Pad the image and generate random background
        X = np.pad(img_np, ((0, 0), (k_len, k_len), (k_len, k_len)), 'constant')
        X11 = np.random.randn(output_depth, d + 2 * k_len, k_len)
        X12 = np.random.randn(output_depth, d + 2 * k_len, k_len)
        X13 = np.random.randn(output_depth, k_len, d)
        X14 = np.random.randn(output_depth, k_len, d)
        
        X[:, :, :k_len] = X11
        X[:, :, -k_len:] = X12
        X[:, :k_len, k_len:-k_len] = X13
        X[:, -k_len:, k_len:-k_len] = X14
        
        # Apply forward model (Fourier magnitude) and Gaussian noise
        img_var = np_to_var(X).type(self.dtype)
        img_var_meas_clean = apply_f(img_var)
        img_var_meas_noisy = add_gaussian_noise(img_var_meas_clean, self.args.noise_std)
        
        return img_var_meas_noisy, (X11, X12, X13, X14)

    def process_single_image(self, img_path):
        """Execute the full experimental loop for a single image across multiple runs."""
        filename = os.path.basename(img_path)
        img_name = os.path.splitext(filename)[0]
        print(f"\n{'='*40}\nProcessing: {filename}\n{'='*40}")

        img_out_dir = os.path.join(self.args.output_dir, f"{img_name}_{self.args.k*2}")
        os.makedirs(img_out_dir, exist_ok=True)
        csv_path = os.path.join(img_out_dir, f"{img_name}_metrics.csv")

        img_np, img_cv2_gt = self.prepare_data(img_path)
        output_depth = img_np.shape[0]
        d = img_np.shape[1]
        num_channels = [128, 128, 128]

        total_psnr, total_ssim = 0.0, 0.0

        with open(csv_path, 'w', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            # Log configurations
            writer.writerow(['Settings:', f"k={self.args.k*2}", f"noise={self.args.noise_std}", 
                             f"iter={self.args.max_iter}", f"lr={self.lr}", f"optim={self.optim_strategy}"])
            writer.writerow(['Run', 'PSNR', 'SSIM', 'Time(s)'])

            for j in range(self.args.runs):
                #set_random_seed(100)
                
                # 1. Simulate measurements
                img_var_meas, (X11, X12, X13, X14) = self.simulate_measurement(img_np, d, output_depth)

                # 2. Initialize the untrained network
                net = autoencodernet(
                    num_output_channels=output_depth, 
                    num_channels_up=num_channels, 
                    need_sigmoid=True, decodetype='upsample'
                ).type(self.dtype)

                # 3. Run the core optimization algorithm (Net-BDR)
                t0 = time.time()
                _, _, _, _, in_np_img, _, _ = fit(
                    net=net, num_channels=num_channels, k=self.args.k, d=d,
                    num_iter=self.args.max_iter, numit_inner=10, 
                    LR=self.lr, OPTIMIZER='adam', lr_decay_epoch=500, 
                    img_clean_var=img_var_meas, find_best=True, code='uniform', 
                    weight_decay=0.000001, decodetype='upsample', 
                    optim=self.optim_strategy, out_channels=output_depth,
                    img_origin=img_np, X1=X11, X2=X12, X3=X13, X4=X14,
                    is_net=self.args.is_net, gamma=self.gamma, 
                    device=self.device, dtype=self.dtype
                )
                t_elapsed = time.time() - t0

                # 4. Calculate metrics and save results
                out_img_np = convert(in_np_img.data.cpu().numpy()[0])[0]
                run_psnr = psnr(out_img_np, img_cv2_gt, data_range=255)
                run_ssim = ssim(out_img_np, img_cv2_gt, data_range=255)
                
                total_psnr += run_psnr
                total_ssim += run_ssim

                print(f"  [Run {j+1}/{self.args.runs}] PSNR: {run_psnr:.2f} | SSIM: {run_ssim:.4f} | Time: {t_elapsed:.1f}s")
                writer.writerow([j+1, f"{run_psnr:.3f}", f"{run_ssim:.4f}", f"{t_elapsed:.2f}"])
                
                cv2.imwrite(os.path.join(img_out_dir, f"{img_name}_run{j+1}_PSNR{run_psnr:.2f}.png"), out_img_np)

            # 5. Summarize and log averages
            avg_psnr, avg_ssim = total_psnr / self.args.runs, total_ssim / self.args.runs
            writer.writerow(['Average', f"{avg_psnr:.3f}", f"{avg_ssim:.4f}", "-"])
            print(f"  => Final Average: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")

    def run(self):
        """Execute the entire evaluation pipeline."""
        image_paths = self.get_image_paths()
        print(f"[System] Found {len(image_paths)} images to process.")
        
        for img_path in image_paths:
            self.process_single_image(img_path)
            
        print("\n[System] All experiments completed successfully.")