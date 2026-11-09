# Real-Time Multi-Modal Vigilance Monitoring (EEG + Facial Features)

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
src/vigilance/
  data/             loading, preprocessing, synchronisation, windowing
  models/           EEG branch, face branch, fusion, Teacher, Student
  training/         training loops, distillation, subject-wise CV
  evaluation/       metrics, statistical tests, robustness, latency
tests/              pytest tests
```
