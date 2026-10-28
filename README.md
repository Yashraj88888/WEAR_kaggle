# WEAR_kaggle

Comparative study of deep learning models for human activity recognition on the
[3rd WEAR Dataset Challenge](https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026)
(ICT 4442 mini project). Each team member owns one model family: MLP, 1D-CNN, BiLSTM and Transformer.

## MLP baseline (Dishank Shah)

Notebook: [`notebooks/mlp_baseline.ipynb`](notebooks/mlp_baseline.ipynb)

### Setup

The task follows the Kaggle test set: one sample is a 1-second window (50 readings at 50 Hz)
of 3-axis acceleration from one of the four limbs, and the label is one of 19 classes
(18 workouts plus `null`).

- **Windows:** 50 samples with a stride of 25. A window is kept only if one label covers at
  least 80% of it, and each window gives one sample per limb. Samples with missing sensor
  readings are dropped.
- **Split:** subjects 0, 1, 8, 13 and 15 are held out for validation, the same subjects as the
  BiLSTM branch, so the two models are scored on the same people. This leaves 412,456
  training and 133,508 validation samples.
- **Input:** the window is normalised per limb and per axis (statistics from the training
  subjects only), flattened to 150 values, and a 4-value one-hot code for the limb is added.
- **Model:** three hidden layers of 512, 256 and 128 units, each with BatchNorm, ReLU and
  dropout 0.3. 247,827 parameters.
- **Training:** AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch 512,
  class-weighted cross-entropy with label smoothing 0.05, early stopping on validation
  macro F1 (patience 6). It stopped after 25 epochs, about 9 minutes on a CPU.

### Validation results

| Metric | Score |
| --- | ---: |
| Accuracy | 55.04% |
| Macro precision | 53.19% |
| Macro recall | 62.39% |
| **Macro F1 (19 classes)** | **56.21%** |
| Weighted F1 | 55.28% |
| Macro F1 without `null` (18 workouts) | 64.63% |

Per-class scores are in [`results/mlp_classification_report.txt`](results/mlp_classification_report.txt),
and the confusion matrix and training curves are in [`results/`](results/).

### What the results show

- **Most errors are between variants of the same exercise.** Jogging with rotating arms is
  called plain jogging 34% of the time, complex push-ups are called push-ups 35% of the
  time, and complex lunges are called lunges 24% of the time. Shoulder stretches are
  confused with triceps stretches (38%). The variants differ in short arm or torso
  movements, and an MLP that sees the window as an unordered list of 150 numbers has a
  hard time picking those up.
- **`null` is the weakest class on recall (0.40) but strong on precision (0.83).** The loss
  gives `null` a weight of 0.25 because it is 40% of the data. That stops the model from
  predicting `null` for everything, but pushes it to label many `null` windows as an
  exercise. Because `null` is 35% of the validation set, this pulls accuracy below macro F1.
- **No overfitting.** Training F1 stays slightly below validation F1 throughout, because
  dropout is active during training. The model is limited by capacity and by ignoring time
  order, not by memorising the training subjects.
- **All four limbs give similar scores** (macro F1 0.55 to 0.58), and the per-subject macro
  F1 ranges from 0.50 (subject 1) to 0.60 (subject 15).

### Comparison with published numbers

The WEAR paper (Bock et al., IMWUT 2024) reports a best inertial-only F1 of 77.10 with
Attend-and-Discriminate. That number is not directly comparable: the paper uses all four
limbs at once, 2-second windows, leave-one-subject-out cross-validation and a 10-second
smoothing filter on the predictions. Our setup classifies 1-second windows from a single
limb, as the Kaggle test set requires, which is a harder task.

### Running it

1. Download from Kaggle (only the inertial files are needed, about 0.8 GB):
   - `train/inertial_feat/*.csv` into `data/train/inertial_feat/`
   - `test/test_inertial_data.npy` and `test/test_meta_data.csv` into `data/test/`
2. `pip install -r requirements.txt`
3. Run `notebooks/mlp_baseline.ipynb` from the `notebooks/` folder. On Kaggle it finds the
   attached competition data automatically.

The notebook writes metrics and plots to `results/`. Model weights and the Kaggle
submission file go to `outputs/`, which is not committed.
