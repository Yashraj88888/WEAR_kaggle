# 1D-CNN for Human Activity Recognition (WEAR, HASCA 2026)

Individual contribution to the ICT 4442 Deep Learning mini project: a comparative study of deep learning models for wearable-sensor human activity recognition on the **3rd WEAR Dataset Challenge (HASCA 2026)**. This folder contains the **1D convolutional neural network** model, owned by Vasavi A S. Other models in the team comparison (MLP, BiLSTM, hybrid transformer) are in their owners' folders. [Link to team repo / sibling folders]

## Task

Classify 1-second windows of tri-axial accelerometer data into **19 classes** (18 exercise/activity types plus `null`):

`null`, jogging, jogging (rotating arms), jogging (skipping), jogging (sidesteps), jogging (butt-kicks), stretching (triceps), stretching (lunging), stretching (shoulders), stretching (hamstrings), stretching (lumbar rotation), push-ups, push-ups (complex), sit-ups, sit-ups (complex), burpees, lunges, lunges (complex), bench-dips.

## Data and preprocessing

The pipeline is shared by all team members so that results are comparable.

- Source: 3rd WEAR Dataset Challenge (Kaggle), accelerometer CSVs from four body locations (right arm, right leg, left leg, left arm), 50 Hz.
- Windows: 50 samples (1 s), stride 25 (50% overlap); a window's label is the majority label and the window is kept only if at least 80% of samples agree.
- Each window comes from **one** limb, so each time segment yields four samples of shape `50 x 3`.
- Per-channel z-score normalization fitted on training subjects only; NaN/Inf values set to 0.
- **Subject-wise split:** subjects `0, 1, 8, 13, 15` are held out for validation (133,508 windows); no subject appears in both sets.
- Training `null` windows are randomly subsampled to a cap (`MAX_TRAIN_NULL_WINDOWS`): 35,000 in the initial run and [your value] in the final runs. The validation set is untouched.

## Models

### v2 (final): `Wear1DCNNv2`, 212,179 trainable parameters

| Stage | Configuration |
|---|---|
| Conv block 1 | Conv1d(3, 64, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.1) |
| Conv block 2 | Conv1d(64, 128, k=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.1) |
| Conv block 3 | Conv1d(128, 256, k=3) + BatchNorm + ReLU + Dropout(0.1) |
| Pooling | Global average pooling and global max pooling, concatenated (512-d) |
| Location embedding | Embedding(4, 16), concatenated with pooled features (528-d) |
| Classifier | Linear(528, 128) + ReLU + Dropout(0.3) + Linear(128, 19) |

### v1 (baseline): `Wear1DCNN`, 36,691 trainable parameters

Two conv blocks (Conv1d 3-64 k=5, 64-128 k=3, dropout 0.25), global average pooling, location embedding, Linear(144, 64) head.

### Training

| Setting | Initial run (v1) | Final runs (v1 retrained, v2) |
|---|---|---|
| Loss | Cross-entropy, sqrt inverse-frequency class weights, label smoothing 0.05 | Same, weights recomputed on the new training set |
| Optimizer | AdamW, weight decay 1e-4 | AdamW, weight decay 1e-4 |
| Learning rate | 2e-4 | 1e-3 |
| Schedule | Cosine annealing (T_max = 30) | Cosine annealing (T_max = 30) |
| Batch size / grad clip | 128 / 1.0 | 128 / 1.0 |
| Training null windows | 35,000 | [your value] |
| Epochs | Up to 30, early stopping (patience 6); stopped at epoch 25 | Fixed 30, no early stopping |
| Reported model | Best validation macro-F1 checkpoint (epoch 19), seed 42 | Final-epoch model, seeds 42, 43, 44 |

## Results (held-out subjects 0, 1, 8, 13, 15)

| Configuration | Parameters | Accuracy | Macro-P | Macro-R | Macro-F1 |
|---|---|---|---|---|---|
| v1, initial run (1 seed, best epoch) | 36,691 | 0.4825 | 0.4871 | 0.5711 | 0.5054 |
| v1, retrained (3 seeds, mean ± std) | 36,691 | 0.6060 ± 0.0054 | 0.5638 ± 0.0058 | 0.5961 ± 0.0038 | 0.5718 ± 0.0008 |
| v2 (3 seeds, mean ± std) | 212,179 | 0.6111 ± 0.0032 | 0.5907 ± 0.0033 | 0.6411 ± 0.0025 | 0.6028 ± 0.0027 |
| **v2, seed 42 (headline model)** | 212,179 | **0.6132** | **0.5917** | **0.6439** | **0.6050** |

The seed-42 headline row is a second run of the same configuration, so it differs slightly from the seed-42 value in `cnn_experiments_all_runs.csv` (macro-F1 0.6037) because of GPU nondeterminism. Per-class scores are in [`cnn_v2_classification_report.txt`](cnn_v2_classification_report.txt).

![Validation confusion matrix (v2)](cnn_v2_confusion_matrix.png)

### Key observations

- Jogging variants are recognized best (butt-kicks F1 0.853, sidesteps 0.820, skipping 0.814).
- `null` recall improved from 0.29 to 0.55; its remaining errors mostly go to the stretching classes (63% of misclassified null windows), which lowers their precision.
- Base and "complex" variants of the same exercise are still confused, but less than in the initial run. Stretching (shoulders) is the weakest class (F1 0.286).
- v2 overfits mildly (training macro-F1 about 0.09 above validation) and its validation score plateaus after about epoch 20.

### Caveats

- The improvement over the initial run bundles several changes (null cap, learning rate, architecture); no ablation separates them.
- All configuration choices were made on the same held-out subjects that are reported, with no separate test set, so the numbers may be somewhat optimistic.
- Causal explanations of the errors are hypotheses; no per-location analysis was run.

## Repository contents

| File | Description |
|---|---|
| `notebook.py` / `notebook.ipynb` | Full pipeline: indexing, windowing, normalization, initial v1 model and training. [Rename the exported notebook source accordingly] |
| `cnn_improvement_cells.py` | v2 model and the multi-seed experiment runner used for the final results |
| `vasavi_wear_1d_cnn_v2_final.pt` | v2 headline checkpoint (seed 42, final epoch) |
| `vasavi_wear_1d_cnn_best.pt` | Initial v1 run checkpoint (best epoch) |
| `cnn_experiments_all_runs.csv` | Final-epoch metrics for every seed of v1 (retrained) and v2 |
| `cnn_v2_classification_report.txt` | Per-class precision / recall / F1 / support for the headline model |
| `cnn_v2_confusion_matrix.png` | Validation confusion matrix for the headline model |
| `cnn_results.csv`, `cnn_training_history.csv`, `classification_report.txt`, `confusion_matrix.png` | Initial run outputs (kept for comparison) |

The cached window arrays (`vasavi_train_sensor_windows.npy` and `vasavi_val_sensor_windows.npy`, hundreds of MB) are regenerated by the notebook. Do not commit them; add `*.npy` to `.gitignore` or use Git LFS.

## Reproducing

1. Open the notebook in Kaggle with the 3rd WEAR Dataset Challenge (HASCA 2026) competition data attached. The notebook expects it at `/kaggle/input/competitions/3rd-wear-dataset-challenge-hasca-2026`.
2. Dependencies: Python 3, `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`. [Pin the versions printed in your Kaggle environment.]
3. Run the notebook up to the data loaders with `MAX_TRAIN_NULL_WINDOWS = [your value]`, then run `cnn_improvement_cells.py` to train v1 and v2 for seeds 42, 43 and 44.

### Loading the v2 checkpoint

`Wear1DCNNv2` is defined in `cnn_improvement_cells.py`.

```python
import torch

model = Wear1DCNNv2(num_classes=19)
ckpt = torch.load("vasavi_wear_1d_cnn_v2_final.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# sensor: float tensor [B, 50, 3] (normalized); location: long tensor [B]
# location ids: 0 right_arm, 1 right_leg, 2 left_leg, 3 left_arm
logits = model(sensor, location)
```

## Future work

Fuse the four limbs as a 12-channel input, add regularization or augmentation for v2, try dilated or residual convolutions and longer windows, add dedicated `null` handling, and run ablations that separate the null cap, learning rate and architecture changes.

