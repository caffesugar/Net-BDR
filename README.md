# Net-BDR: Untrained Neural Networks embedded Background Douglas Rachford method for Fourier Phase Retrieval

Official PyTorch implementation of the paper: **"Net-BDR: Untrained Neural Networks embedded Background Douglas Rachford method for Fourier Phase Retrieval"** accepted by *IEEE Transactions on Computational Imaging (TCI)*.

## 📖 Introduction
Background Douglas-Rachford (BDR) methods incorporate background information as a structural prior to alleviate the ill-posedness of Fourier Phase Retrieval (FPR). However, traditional methods often suffer from poor robustness against noise.

**Net-BDR** integrates the implicit prior of Untrained Neural Networks (UNN) with the structural prior of the background. Our method:
* **Improves robustness** against high-level noise.
* **Reduces the requirement** for precise background information.
* **Accelerates computation** through an efficient optimization strategy.
* **Theoretical Guarantee**: Confirms convergence via rigorous mathematical analysis.

**Paper DOI:** [10.1109/TCI.2026.3670621](https://doi.org/10.1109/TCI.2026.3670621)

---

## 🖼️ Framework
![Net-BDR Pipeline](framework.png)

---

## 🚀 Setup & Installation

We recommend using **Conda** for environment isolation and Python version consistency (Python 3.10.12).

### Option 1: Conda (Recommended)
```bash
# Create the environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate Net-BDR
```

### Option 2: Pip
```bash
# Ensure you have Python 3.10+ installed
pip install -r requirements.txt
```

---

## 🛠️ Usage

### 1. Data Preparation
Place your test images (e.g., `Cameraman.png`) in the `./test_images` directory. The images should be square (e.g., 128x128 or 256x256).

### 2. Running the Code
You can run the phase retrieval process using `main.py`. Below are examples of common execution commands:

**Basic Run:**
```bash
python main.py --k 0.2 --max_iter 300 --runs 5
```

**Run with Noise Simulation:**
```bash
python main.py --noise_std 0.01 --k 0.3 --max_iter 300
```

### Argument Details:
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--k` | Padding ratio for background information(k*2) | `0.2` |
| `--noise_std` | Standard deviation of additive Gaussian noise | `0.0` |
| `--max_iter` | Total number of optimization iterations | `300` |
| `--runs` | Number of independent trials for averaging | `5` |
| `--gpu_id` | Specify which GPU to use (e.g., 0, 1) | `0` |

---

## 📊 Results
Reconstruction results and performance metrics (PSNR, SSIM) are automatically saved in the `./results` directory. Each image test creates a sub-folder containing:
* `metrics.csv`: Convergence data and final evaluation.
* `*.png`: Visual reconstruction results for each run.

---

## 📝 Citation
If you find this code or our research useful, please cite our IEEE TCI paper:

```bibtex
@ARTICLE{11421092,
  author={Yang, Yi and Ma, Liyuan and Yuan, Ziyang and Wang, Hongxia},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Net-BDR: Untrained Neural Networks Embedded Background Douglas Rachford Method for Fourier Phase Retrieval}, 
  year={2026},
  volume={12},
  number={},
  pages={673-688},
  keywords={Noise;Image reconstruction;Robustness;Neural networks;Convergence;Phase measurement;Imaging;Noise measurement;Loss measurement;Signal processing algorithms;Fourier phase retrieval (FPR);untrained neural networks;background information;Douglas Rachford method},
  doi={10.1109/TCI.2026.3670621}}
```

## Contact
**Yang Yi** (yangyi24@nudt.edu.cn)  
National University of Defense Technology (NUDT)
