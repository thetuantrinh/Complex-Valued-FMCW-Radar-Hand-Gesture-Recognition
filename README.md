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

### [📄 Paper (IEEE TAES)](https://doi.org/10.1109/TAES.2026.3709269) • [💡 Key Highlights](#-key-breakthroughs--contributions) • [🧠 Model Architecture](#-21d-cvnet-architecture) • [📁 Repository](#-repository-structure) • [🚀 Quick Start](#-quick-start) • [🧪 Development](#-development) • [📖 Citation](#-citation)

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
   │ (2+1)D Complex-Valued Convolution (complex_conv_2plus1d)                       │
   │  ├─ Spatial Decomposition: ComplexConv3D with Kernel (1, K_chirp, K_sample)    │
   │  └─ Temporal Decomposition: ComplexConv3D with Kernel (K_time, 1, 1)           │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ Complex Batch Normalization (ComplexBatchNormalization) + Complex ReLU         │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ 4× Residual Blocks with Complex Identity / Projection Shortcuts                │
   │  ├─ Residual Block 1 & 2: 4 Complex Filters (Kernel: 3×3×2)                    │
   │  └─ Residual Block 3 & 4: 8 Complex Filters (Kernel: 3×3×2)                    │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ 3D Max-Pooling + Global Average Pooling 3D                                     │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ Monte Carlo Dropout (p = 0.05, training=True for Epistemic Sampling)           │
   └────────────────────────────────────┬───────────────────────────────────────────┘
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────────┐
   │ Complex Dense Classifier ──► Softmax Probabilities & Uncertainty Variance      │
   └────────────────────────────────────────────────────────────────────────────────┘
```

The core $(2+1)\text{D}$ decomposition factors a full complex 3D convolution of size $K_t \times K_r \times K_d$ into:
1. An intermediate **spatial convolution** $(1 \times K_r \times K_d)$ operating across the fast-time sample and chirp domains.
2. A subsequent **temporal convolution** $(K_t \times 1 \times 1)$ operating across successive radar frames.

This separation drastically reduces the required parameter count and multiplication operations while capturing both intra-frame radar phase characteristics and inter-frame temporal kinematics.

---

## 📂 Repository Structure

```text
Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition/
├── src/cvradar/                      # Installable Python package
│   ├── config.py                     # Typed, validated experiment configuration
│   ├── cli.py                        # `cvradar {train,evaluate,summary}` entry point
│   ├── data/
│   │   ├── labels.py                 # Canonical gesture vocabulary & label encoding
│   │   └── loaders.py                # tf.data pipelines (clean + noise-corrupted)
│   ├── models/
│   │   ├── layers.py                 # Complex-valued (2+1)D convolution & residual blocks
│   │   └── cv_net.py                 # (2+1)D CVNet architecture
│   ├── training/
│   │   ├── callbacks.py              # LR schedule, checkpointing, TensorBoard
│   │   └── trainer.py                # Training loop & run artifacts
│   ├── evaluation/
│   │   ├── metrics.py                # Expected Calibration Error & NLL
│   │   ├── uncertainty.py            # MC Dropout, Deep Ensembles, entropy decomposition
│   │   └── evaluate.py               # Evaluation driver
│   └── viz/
│       └── plots.py                  # Confusion matrix, training curves, reliability diagram
├── configs/
│   ├── default.yaml                  # Published configuration
│   └── deep_ensemble.yaml            # Deep Ensemble evaluation preset
├── checkpoints/
│   ├── deep_ensemble/                # 10 ensemble members for epistemic variance
│   └── mc_dropout/                   # Monte Carlo Dropout model
├── tests/                            # pytest suite (TF-dependent tests skip cleanly)
├── data/
│   └── README.md                     # Acquisition protocol & subject partitioning
├── docs/assets/                      # Architecture diagram
├── .github/workflows/ci.yml          # Lint + test continuous integration
├── pyproject.toml                    # Package metadata, ruff, mypy & pytest config
├── requirements.txt                  # Pinned runtime dependencies
├── LICENSE                           # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation

The code targets **Python 3.8–3.10** (an upper bound inherited from
`tensorflow < 2.16`, which `keras-complex` requires).

```bash
git clone https://github.com/thetuantrinh/Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition.git
cd Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition

conda create -n cvradar python=3.9 -y
conda activate cvradar

pip install -e .            # add [dev] for the test and lint tooling
```

### 2. Point the code at your dataset

Nothing in the package hardcodes a dataset path. Set the root once:

```bash
export CVRADAR_DATA_ROOT=/path/to/dataset
```

…or set `data.root` in a config file, or pass `--data-root` on the command line.
The expected layout is `<root>/<split>/<subject>/<gesture>/<sample>.npy`; see
[data/README.md](data/README.md).

### 3. Command line

```bash
# Inspect the architecture and parameter count
cvradar summary

# Train from scratch
cvradar train --config configs/default.yaml

# Evaluate the released Monte Carlo Dropout model
cvradar evaluate --model checkpoints/mc_dropout/run_20250510_101611

# Evaluate the Deep Ensemble under -5 dB AWGN
cvradar evaluate --config configs/deep_ensemble.yaml \
                 --split noise --noise-type AWGN_SNR_-5
```

Every run writes its SavedModel, resolved configuration and training history
into a timestamped directory under `checkpoints/`.

### 4. Python API

```python
from cvradar.config import ExperimentConfig
from cvradar.models.cv_net import build_cv_net

config = ExperimentConfig.from_yaml("configs/default.yaml")

# Input: [Batch, Frames (20), Chirps (128), Samples (64), Channels (8)]
# Channels = 4 RX antennas x 2 (real / imaginary)
model = build_cv_net(config.radar, config.model)
model.summary()
```

Uncertainty-aware inference with the released checkpoints:

```python
import tensorflow as tf

from cvradar.data.loaders import load_split
from cvradar.evaluation.metrics import UncertaintyMetrics
from cvradar.evaluation.uncertainty import mc_dropout_predict, mutual_information

model = tf.keras.models.load_model("checkpoints/mc_dropout/run_20250510_101611")
dataset = load_split("valid", config.data, config.radar)

features, labels = next(iter(dataset))
mean_probs, samples = mc_dropout_predict(model, features, num_samples=50)

metrics = UncertaintyMetrics(tf.math.log(mean_probs), labels)
print(metrics.summary())  # accuracy, ECE, NLL
print("epistemic:", float(tf.reduce_mean(mutual_information(samples))))
```

---

## 🧪 Development

```bash
pip install -e ".[dev]"

pytest                 # test suite; TF-dependent tests skip if TF is absent
ruff check .           # lint
ruff format .          # format
mypy                   # type check
```

---

## 📊 Dataset Access

The experimental dataset contains raw FMCW radar time-domain ADC recordings captured using a **Texas Instruments AWR1243BOOST** 77 GHz mmWave sensor coupled with a **DCA1000EVM** high-speed capture card:
- **10 Gesture Classes** performed by **10 volunteers**.
- Partitioned into **Training** (Subjects 1–8), **Clean Validation** (Subjects 9–10), and **Noise Validation** (Subjects 9–10 corrupted with synthetic AWGN from $-5\text{ dB}$ to $+10\text{ dB}$).

> [!NOTE]  
> Due to storage capacity considerations, the raw dataset archives are available upon academic request.  
> Please contact **Dr. Minhhuy Le** ([huy.leminh@phenikaa-uni.edu.vn](mailto:huy.leminh@phenikaa-uni.edu.vn)) or consult [data/README.md](data/README.md).

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
