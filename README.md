<div align="center">

# Complex-Valued (2+1)D Convolutional Neural Networks<br>for Real-Time Hand Gesture Recognition on Edge Devices With FMCW Radar

[![IEEE TAES](https://img.shields.io/badge/IEEE-Transactions_on_Aerospace_and_Electronic_Systems_2026-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11593416/)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTAES.2026.3709269-0288D1?style=for-the-badge)](https://doi.org/10.1109/TAES.2026.3709269)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[![Accuracy](https://img.shields.io/badge/Accuracy-99.38%25-brightgreen?style=flat-square&logo=target&logoColor=white)]()
[![Edge Latency](https://img.shields.io/badge/NVIDIA_Jetson_Nano-2.75_ms-76B900?style=flat-square&logo=nvidia&logoColor=white)]()
[![Speedup](https://img.shields.io/badge/Speedup-4×_to_86×_Faster-blueviolet?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.8_|_3.9_|_3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Complex-Valued](https://img.shields.io/badge/Network-Complex--Valued_CNN-1565C0?style=flat-square)]()

<br>

**Official repository for the research article:**  
*"Complex-Valued (2+1)D Convolutional Neural Networks for Real-Time Hand Gesture Recognition on Edge Devices With FMCW Radar"*  
Published in **IEEE Transactions on Aerospace and Electronic Systems (TAES)**, Vol. 62, pp. 13147–13156, 2026.  
DOI: [10.1109/TAES.2026.3709269](https://doi.org/10.1109/TAES.2026.3709269)

[**The Tuan Trinh**](https://github.com/thetuantrinh)$^1$ · [**Phan Xuan Tan**](https://orcid.org/0000-0002-9592-0226)$^2$ · [**Khai Nguyen Van**](https://orcid.org/0009-0007-2802-541X)$^1$ · [**Pham Vu Bao Tram**](https://orcid.org/0009-0009-7924-5484)$^1$ · [**Xuanque Nguyen**](https://orcid.org/0009-0001-2635-6553)$^3$ · [**Khoa Nguyen Dang**](https://orcid.org/0000-0002-6525-5245)$^4$ · [**Minhhuy Le**](https://orcid.org/0000-0001-6152-6215)$^{1,*}$

*$^1$ Faculty of Electrical and Electronic Engineering, Phenikaa School of Engineering, Phenikaa University, Hanoi, Vietnam*  
*$^2$ College of Engineering, Shibaura Institute of Technology, Tokyo, Japan*  
*$^3$ Faculty of Information System, Phenikaa School of Computing, Phenikaa University, Hanoi, Vietnam*  
*$^4$ Faculty of Engineering and Technology, International School, Vietnam National University (VNU-IS), Hanoi, Vietnam*  
$^*$*Corresponding author*: [huy.leminh@phenikaa-uni.edu.vn](mailto:huy.leminh@phenikaa-uni.edu.vn)

---

### [📄 Paper (IEEE TAES)](https://doi.org/10.1109/TAES.2026.3709269) • [💡 Key Highlights](#-key-breakthroughs--contributions) • [🧠 Model Architecture](#-21d-cvnet-architecture) • [📊 Quantitative Results](#-benchmarks--edge-performance) • [🚀 Quick Start](#-quick-start) • [📖 Citation](#-citation)

---

</div>

## 📖 Abstract

Radar-based hand gesture recognition has emerged as a key enabler for intuitive and privacy-preserving human–machine interaction. However, most existing methods rely on computationally intensive frequency-domain preprocessing—such as the 2D/3D Fast Fourier Transform (FFT)—which is time-consuming and creates severe latency bottlenecks on resource-constrained embedded edge devices.

This article introduces **(2+1)D Complex-Valued Network ((2+1)D CVNet)**, a paradigm-shifting architecture that directly learns discriminative spatio-temporal representations from **raw time-domain Frequency-Modulated Continuous-Wave (FMCW) radar signals**. By eliminating the FFT transformation stages entirely, the proposed framework performs end-to-end learning in the **native complex domain ($\mathbb{C}$)**, drastically reducing computational overhead while retaining critical RF phase and amplitude correlations. To guarantee trustworthy decision-making in noisy and dynamic environments, **Monte Carlo Dropout (MC-D)** and **Deep Ensemble Learning (DEL)** are integrated for robust epistemic uncertainty quantification.

Extensive experimental evaluations demonstrate that **(2+1)D CVNet achieves 99.38% classification accuracy** across 10 gesture classes, executes **$4\times$ to $86\times$ faster** than FFT-based counterparts, runs in **only 2.75 ms** on the resource-constrained **NVIDIA Jetson Nano**, and maintains superior noise resilience down to **$-5\text{ dB}$ SNR**.

---

## 💡 Key Breakthroughs & Contributions

- 🚀 **FFT-Free End-to-End Learning**: Operates directly on raw ADC in-phase and quadrature ($I/Q$) time-domain signals, bypassing expensive Range-Doppler and Micro-Doppler FFT matrix generation.
- ⚡ **Ultra-Low Edge Latency**: Achieves an inference latency of **2.75 ms on the NVIDIA Jetson Nano**, enabling real-time edge AI human-machine interaction at >360 FPS.
- ⏱️ **$4\times$ to $86\times$ Speedup**: Outperforms conventional 3D CNNs and hybrid CNN-LSTM pipelines in throughput while reducing parameter count and memory footprint.
- 🎯 **99.38% Accuracy**: Validated on dynamic multi-user radar data across 10 distinct gesture classes.
- 🛡️ **Severe Noise Robustness**: Resilient down to **$-5\text{ dB}$ SNR** under additive Gaussian noise and environmental clutter.
- 📐 **Calibrated Uncertainty Estimation**: Integrates Monte Carlo Dropout (MC-D) and Deep Ensemble Learning (DEL) with rigorous validation via **Expected Calibration Error (ECE)** and **Negative Log-Likelihood (NLL)**.

---

## 🧠 (2+1)D CVNet Architecture

```
   Raw FMCW Time-Domain Radar Tensor [Batch, Time (20), Chirps (128), Samples (64), Channels (8)]
                                        │
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ (2+1)D Complex-Valued Convolution (CV_Conv2Plus1D)                             │
   │  ├─ Spatial Decomposition: ComplexConv3D with Kernel (1, K_chirp, K_sample)   │
   │  └─ Temporal Decomposition: ComplexConv3D with Kernel (K_time, 1, 1)          │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ Complex Batch Normalization (ComplexBatchNormalization) + Complex ReLU        │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ 4× Residual Blocks with Complex Identity / Projection Shortcuts                │
   │  ├─ Residual Block 1 & 2: 4 Complex Filters (Kernel: 3×3×2)                   │
   │  └─ Residual Block 3 & 4: 8 Complex Filters (Kernel: 3×3×2)                   │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ 3D Max-Pooling + Global Average Pooling 3D                                     │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ Monte Carlo Dropout (p = 0.05, training=True for Epistemic Sampling)          │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ Complex Dense Classifier ──► Softmax Probabilities & Uncertainty Variance     │
   └────────────────────────────────────────────────────────────────────────────────┘
```

The core $(2+1)\text{D}$ decomposition factors a full complex 3D convolution of size $K_t \times K_r \times K_d$ into:
1. An intermediate **spatial convolution** $(1 \times K_r \times K_d)$ operating across the fast-time sample and chirp domains.
2. A subsequent **temporal convolution** $(K_t \times 1 \times 1)$ operating across successive radar frames.

This separation drastically reduces the required parameter count and multiplication operations while capturing both intra-frame radar phase characteristics and inter-frame temporal kinematics.

---

## 📊 Benchmarks & Edge Performance

### Comparison with Benchmark Architectures

| Model Architecture | Input Domain | Preprocessing Latency | Inference Latency (Jetson Nano) | Total Latency | Accuracy (%) | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conventional 3D CNN** | FFT (Range-Doppler) | ~35.4 ms | ~28.6 ms | ~64.0 ms | 98.12% | $1.0\times$ (Baseline) |
| **2D CNN + LSTM** | FFT (Micro-Doppler) | ~22.1 ms | ~14.8 ms | ~36.9 ms | 97.45% | $1.7\times$ |
| **Real-Valued (2+1)D CNN** | Raw Time-Domain | **0.0 ms** | 4.85 ms | 4.85 ms | 96.20% | $13.2\times$ |
| **Proposed (2+1)D CVNet** | **Raw Time-Domain** | **0.0 ms (FFT-Free)** | **2.75 ms** | **2.75 ms** | **99.38%** | **$23.3\times$ – $86\times$** |

### Noise Resilience (Classification Accuracy vs. SNR)

| Signal-to-Noise Ratio (SNR) | Clean | $+10\text{ dB}$ | $+5\text{ dB}$ | $0\text{ dB}$ | $-5\text{ dB}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conventional FFT 3D CNN** | 98.12% | 94.30% | 88.50% | 79.20% | 68.40% |
| **Proposed (2+1)D CVNet** | **99.38%** | **98.80%** | **97.65%** | **95.20%** | **92.15%** |

---

## 🖐️ Gesture Vocabulary

The dataset encompasses 10 distinct dynamic gesture classes evaluated under clean and noise-perturbed settings:

| Class ID | Gesture Name | Motion Description | Kinetic Axis |
| :-: | :--- | :--- | :--- |
| **0** | `Push-down` | Rapid downward palm thrust | Vertical ($Z^-$) |
| **1** | `Pull-up` | Rapid upward palm lift | Vertical ($Z^+$) |
| **2** | `Counter-Clockwise` | Circular hand rotation in CCW direction | Angular ($\theta^+$) |
| **3** | `Clockwise` | Circular hand rotation in CW direction | Angular ($\theta^-$) |
| **4** | `Zoom-in` | Two-finger / dual-palm contracting squeeze | Radial inward ($r^-$) |
| **5** | `Zoom-out` | Dual-palm expanding outward spread | Radial outward ($r^+$) |
| **6** | `To-left` | Horizontal swipe from right to left | Lateral ($X^+$) |
| **7** | `To-right` | Horizontal swipe from left to right | Lateral ($X^-$) |
| **8** | `No gesture` | Stationary hand / idle ambient background | Null |
| **9** | `Unknown gesture` | Out-of-distribution / irregular gestures | OOD |

---

## 📂 Repository Structure

```text
Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition/
├── Datasets/
│   └── README.md                      # Dataset acquisition protocol & subject partition details
├── scripts/
│   ├── network/
│   │   ├── CV_3DNet.py                # Native (2+1)D CVNet architecture & Complex-Valued layers
│   │   ├── CV_3DNet_Structure.png     # Graphical network architecture diagram
│   │   └── __init__.py
│   ├── models/                        # Pre-trained model weights
│   │   ├── Deep Ensembles Learing/    # 10 ensemble model checkpoints for epistemic variance
│   │   └── MC-D/                      # Monte Carlo Dropout model checkpoint
│   ├── utils/
│   │   ├── load_dataset.py            # Clean validation dataset loader
│   │   ├── load_noise_dataset.py      # Noise-corrupted dataset loader (AWGN SNR validation)
│   │   ├── Uncertainty_Metrics.py     # Expected Calibration Error (ECE) & Negative Log-Likelihood
│   │   ├── utils_plots.py             # Confusion matrix & accuracy curves visualization
│   │   ├── callbacks.py               # Training callbacks and learning rate schedules
│   │   └── __init__.py
│   └── requirements.txt               # Dependency specifications (TensorFlow, Keras-Complex, etc.)
├── .gitignore
├── LICENSE                            # MIT License
└── README.md                          # Primary documentation
```

---

## 🚀 Quick Start

### 1. Environment Setup

Clone this repository and set up a Python 3.8–3.10 virtual environment:

```bash
git clone https://github.com/thetuantrinh/Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition.git
cd Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition

# Create virtual environment
conda create -n cv_radar python=3.9 -y
conda activate cv_radar

# Install dependencies
pip install -r scripts/requirements.txt
```

### 2. Instantiate the (2+1)D CVNet Model

```python
import sys
sys.path.insert(0, "scripts")

from network.CV_3DNet import CV_Net

# Input format: [Batch, Time (20), Chirps (128), Samples (64), Channels (8)]
# Channels (8) = 4 RX antennas * 2 (Real/Imaginary components)
model = CV_Net(input_shape=[None, 20, 128, 64, 8], output_shape=10)
model.summary()
```

### 3. Evaluating Pretrained Checkpoints with Uncertainty

```python
import tensorflow as tf
from utils.Uncertainty_Metrics import Uncertainty_Metrics
from utils.load_dataset import load_data

# Load clean validation dataset
val_loader = load_data(batch_size=32)

# Load pre-trained Monte Carlo Dropout model
model = tf.keras.models.load_model("scripts/models/MC-D/20250510_101611")

# Compute predictions and calibration metrics
for x_val, y_val in val_loader.take(1):
    logits = model(x_val, training=True)
    metrics = Uncertainty_Metrics(x_val, y_val, logits)
    print(f"Expected Calibration Error (ECE): {metrics._ECE().numpy():.4f}")
    print(f"Negative Log-Likelihood (NLL):    {metrics._NLL().numpy():.4f}")
```

---

## 📊 Dataset Access

The experimental dataset contains raw FMCW radar time-domain ADC recordings captured using a **Texas Instruments AWR1243BOOST** 77 GHz mmWave sensor coupled with a **DCA1000EVM** high-speed capture card:
- **10 Gesture Classes** performed by **10 volunteers**.
- Partitioned into **Training** (Subjects 1–8), **Clean Validation** (Subjects 9–10), and **Noise Validation** (Subjects 9–10 corrupted with synthetic AWGN from $-5\text{ dB}$ to $+10\text{ dB}$).

> [!NOTE]  
> Due to storage capacity considerations, the raw dataset archives are available upon academic request.  
> Please contact **Dr. Minhhuy Le** ([huy.leminh@phenikaa-uni.edu.vn](mailto:huy.leminh@phenikaa-uni.edu.vn)) or consult [Datasets/README.md](Datasets/README.md).

---

## 📖 Citation

If you use this repository, the (2+1)D CVNet architecture, or the radar gesture dataset in your research, please cite our IEEE TAES paper:

```bibtex
@article{trinh2026complex,
  author={Trinh, The Tuan and Tan, Phan Xuan and Van, Khai Nguyen and Tram, Pham Vu Bao and Nguyen, Xuanque and Dang, Khoa Nguyen and Le, Minhhuy},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={Complex-Valued (2+1)D Convolutional Neural Networks for Real-Time Hand Gesture Recognition on Edge Devices With FMCW Radar}, 
  year={2026},
  volume={62},
  pages={13147--13156},
  doi={10.1109/TAES.2026.3709269}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete terms.

---

## 🏛️ Institutional Acknowledgements

This research was developed through collaboration among:
- **Intelligent Communication System Laboratory (ICSLab)**, Phenikaa School of Engineering, Phenikaa University, Hanoi, Vietnam.
- **College of Engineering**, Shibaura Institute of Technology, Tokyo, Japan.
- **Faculty of Information System**, Phenikaa School of Computing, Phenikaa University, Hanoi, Vietnam.
- **Faculty of Engineering and Technology**, International School, Vietnam National University (VNU-IS), Hanoi, Vietnam.
