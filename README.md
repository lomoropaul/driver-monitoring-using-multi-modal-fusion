# Driver Monitoring Using Modal Fusion

Final-year project, BSc Computer Science (AI & Robotics), Uganda Christian University.

A deep learning framework that fuses EEG signals with facial features to detect driver
vigilance decline in 4-second windows, then compresses the model by knowledge distillation
for real-time use on an NVIDIA Jetson Orin Nano.

## Pipeline

1. **EEG branch:** compact spatio-temporal CNN (EEGNet-style)
2. **Face branch:** lightweight CNN on eye/mouth crops, GRU over frames, plus EAR, MAR and PERCLOS
3. **Fusion:** early (feature concatenation), late (weighted probabilities), hybrid (attention gate)
4. **Teacher → Student:** knowledge distillation, target ≥70% smaller
5. **Deployment:** ONNX, then TensorRT on Jetson (target <100 ms per window)

## Datasets

| Dataset | Modalities | Use |
|---|---|---|
| DROZY | EEG + NIR face video (synchronised) | All fusion and distillation experiments |
| SEED-VIG | EEG | EEG branch pre-training |
| NTHU-DDD | Face video | Face branch pre-training, occlusion tests |

Datasets are not included in this repository. Place them under `data/raw/<dataset_name>/`.

## Setup (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe scripts\check_env.py
```

## Layout

```
configs/            experiment settings (default.yaml)
data/raw/           downloaded datasets (git-ignored)
data/processed/     windowed, preprocessed arrays (git-ignored)
scripts/            command-line entry points
src/driver_monitoring/
  data/             loading, preprocessing, synchronisation, windowing
  models/           EEG branch, face branch, fusion, Teacher, Student
  training/         training loops, distillation, subject-wise CV
  evaluation/       metrics, statistical tests, robustness, latency
tests/              pytest tests
```

## Running the experiments (DROZY)

DROZY is available for research use only and may not be redistributed; request it from
the authors and cite Massoz et al. (2016). Unzip it to `data/raw/drozy/`, then:

```powershell
# 1. Preprocess EEG, align video to EEG, extract face crops and features
.venv\Scripts\python.exe scripts\prepare_drozy.py
.venv\Scripts\python.exe scripts\prepare_drozy.py --face

# 2. Single-modality baselines (leave-one-subject-out, 14 folds)
.venv\Scripts\python.exe scripts\train_baseline.py --model eeg
.venv\Scripts\python.exe scripts\train_baseline.py --model face

# 3. Fusion (late fusion reuses the baselines)
.venv\Scripts\python.exe scripts\train_fusion.py --strategy late
.venv\Scripts\python.exe scripts\train_fusion.py --strategy early
.venv\Scripts\python.exe scripts\train_fusion.py --strategy hybrid

# 4. Teacher, Student without distillation (ablation), distilled Student
.venv\Scripts\python.exe scripts\train_distill.py --stage teacher
.venv\Scripts\python.exe scripts\train_distill.py --stage scratch
.venv\Scripts\python.exe scripts\train_distill.py --stage student

# 5. Results table, size, CPU latency and significance tests (H1, H2)
.venv\Scripts\python.exe scripts\compare_runs.py
```

Each run writes predictions, per-fold metrics and a summary to `runs/<name>/`.
Training resumes from the last finished fold if interrupted, and `--folds` plus
`--threads` let several processes share the folds.

Run the tests with `.venv\Scripts\python.exe -m pytest`.

## Training on Google Colab

`notebooks/colab_train.ipynb` runs the experiments on a Colab GPU. Training uses the
GPU automatically when one is available (`--device auto`). Upload the processed data
(`data/processed/` plus `runs/teacher/`, zipped as `drozy_processed.zip`) to a private
Google Drive folder `MyDrive/driver-monitoring/`; results are saved to Drive after
every fold, so a disconnected session resumes where it stopped.

## Testing a trained model on one driver

```powershell
.venv\Scripts\python.exe scripts\test_model.py --subject 7                       # Teacher
.venv\Scripts\python.exe scripts\test_model.py --subject 7 --model student_kd    # distilled Student
.venv\Scripts\python.exe scripts\test_model.py --subject 4 --test 3 --model early_fusion
```

It uses the model from the fold in which that driver was held out, so the driver
was never seen in training. It prints, per session, the sleepiness rating, the
true state, how many 4-second windows were flagged as vigilance decline and the
accuracy, and saves a plot of the decline probability over time to `runs/tests/`.

## Edge deployment (Jetson Orin Nano)

```powershell
# Export a fusion model to ONNX (fold in which driver 7 was held out), checked against PyTorch
.venv\Scripts\python.exe scripts\export_onnx.py --model teacher --subject 7
.venv\Scripts\python.exe scripts\export_onnx.py --model student_kd --subject 7

# Run the in-vehicle loop on a recorded session: one 4 s window at a time,
# 1-minute averaging, OK / WARNING (0.5) / STOP (0.8), with per-window latency
.venv\Scripts\python.exe scripts\edge_demo.py --subject 7 --test 3 --model teacher
```

On a Jetson the same `.onnx` file runs with ONNX Runtime (CUDA/TensorRT providers
are used automatically when installed) or can be compiled with
`trtexec --onnx=exports/teacher_S7.onnx --fp16 --saveEngine=teacher_S7.engine`.
