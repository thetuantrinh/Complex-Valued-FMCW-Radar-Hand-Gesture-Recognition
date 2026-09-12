# Dataset

Raw time-domain FMCW radar recordings for the IEEE TAES article
*"Complex-Valued (2+1)D Convolutional Neural Networks for Real-Time Hand Gesture
Recognition on Edge Devices With FMCW Radar"*.

The recordings are **not** committed to this repository. They are available on
academic request from the corresponding author,
[Dr. Minhhuy Le](mailto:huy.leminh@phenikaa-uni.edu.vn).

## Acquisition

| Property | Value |
| --- | --- |
| Sensor | Texas Instruments [AWR1243BOOST](https://www.ti.com/tool/AWR1243BOOST), 77 GHz mmWave |
| Capture card | Texas Instruments [DCA1000EVM](https://www.ti.com/tool/DCA1000EVM) |
| Gesture classes | 10 |
| Volunteers | 10 |
| Frames per sample | 20 |
| Chirps per frame | 128 |
| ADC samples per chirp | 64 |
| Receive antennas | 4 (I and Q each → 8 channels) |

Each `.npy` file holds one gesture as a `float32` array of shape
`(20, 128, 64, 8)`. **No FFT is applied** — the network consumes the raw
in-phase/quadrature time-domain tensor directly.

## Subject partitioning

Splits are subject-disjoint, so validation measures generalisation to unseen
people rather than to unseen repetitions.

| Split | Subjects | Purpose |
| --- | --- | --- |
| `train` | Person 1–8 | Model fitting |
| `valid` | Person 9–10 | Clean validation |
| `noise` | Person 9–10 | AWGN-corrupted validation, −5 dB to +10 dB SNR |

## Expected directory layout

The loaders in `cvradar.data.loaders` glob `<split>/<subject>/<gesture>/*.npy`.
Point the code at the dataset root with `$CVRADAR_DATA_ROOT`, the `data.root`
config key, or `--data-root`:

```text
$CVRADAR_DATA_ROOT/
├── train/
│   ├── Person_1/
│   │   ├── clock-wise/
│   │   │   ├── data_1.npy
│   │   │   └── ...
│   │   ├── counter_clock-wise/
│   │   ├── empty/
│   │   ├── pull-up/
│   │   ├── push-down/
│   │   ├── to_left/
│   │   ├── to_right/
│   │   ├── unknown/
│   │   ├── zoom-in/
│   │   └── zoom-out/
│   ├── Person_2/
│   └── ...  (through Person_8)
├── valid/
│   ├── Person_9/
│   └── Person_10/
└── noise/
    ├── AWGN_SNR_-5/
    │   ├── Person_9/
    │   └── Person_10/
    ├── AWGN_SNR_0/
    ├── AWGN_SNR_5/
    └── AWGN_SNR_10/
```

The split directory names are configurable (`data.train_subdir`,
`data.valid_subdir`, `data.noise_subdir`), but the **gesture directory names
above are not**: they define the class-index order in
`cvradar.data.labels.GESTURE_CLASSES`, which the released checkpoints depend on.
A gesture directory outside that vocabulary raises an error rather than being
silently assigned an index.
