# Net-BDR: Untrained Neural Networks embedded Background Douglas Rachford method for Fourier Phase Retrieval

Official implementation of the paper accepted by **IEEE Transactions on Computational Imaging**.

## 📖 Introduction
Background Douglas-Rachford (BDR) methods incorporate background information as a structural prior to alleviate the ill-posedness of Fourier Phase Retrieval (FPR). However, traditional methods suffer from poor robustness. 

**Net-BDR** integrates the implicit prior of Untrained Neural Networks (UNN) with the structural prior of the background. Our method:
* **Improves robustness** against noise.
* **Reduces the requirement** for background information.
* **Accelerates computation** through an efficient strategy.

**DOI:** [10.1109/TCI.2026.3670621](https://doi.org/10.1109/TCI.2026.3670621)

## 🖼️ Framework
![Net-BDR Pipeline](framework.png)

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/caffesugar/Net-BDR.git](https://github.com/caffesugar/Net-BDR.git)
cd Net-BDR
pip install -r requirements.txt