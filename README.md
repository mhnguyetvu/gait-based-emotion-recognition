# STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits

This repository contains the official implementation of the paper:

STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits

Reference:

```bibtex
@inproceedings{bhattacharya2020step,
author = {Bhattacharya, Uttaran and Mittal, Trisha and Chandra, Rohan and Randhavane, Tanmay and Bera, Aniket and Manocha, Dinesh},
title = {STEP: Spatial Temporal Graph Convolutional Networks for Emotion Perception from Gaits},
year = {2020},
publisher = {AAAI Press},
booktitle = {Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
pages = {1342--1350},
numpages = {9},
series = {AAAI'20}
}
```

The project is a research pipeline for gait-based emotion recognition and includes a CVAE generator, multiple ST-GCN classifiers, and affective-feature extraction.

## Project structure

- `generator_cvae/` — generates synthetic gait samples
- `classifier_stgcn_real_only/` — baseline classifier trained on real data only
- `classifier_stgcn_real_and_synth/` — baseline classifier trained on real + synthetic data
- `classifier_hybrid/` — hybrid model using deep and physiological features
- `compute_aff_features/` — computes affective features from 16-joint pose sequences
- `torchlight/` — local PyTorch utility package used by the project
- `requirements.txt` — pinned dependency list

## What you need before running

This repo expects a dataset in a top-level `data/` folder. The dataset is not included in this repository. The paper mentions the Emotion-Gait dataset, which should be downloaded from the official release page.

The scripts load files such as:

- `features.h5`
- `labels.h5`
- `features4DCVAEGCN.h5`
- `labels4DCVAEGCN.h5`
- `affectiveFeatures*.h5`

The expected folder layout is:

```text
STEP/
  README.md
  requirements.txt
  torchlight/
  generator_cvae/
  classifier_stgcn_real_only/
  classifier_stgcn_real_and_synth/
  classifier_hybrid/
  compute_aff_features/
  data/
    features.h5
    labels.h5
    features4DCVAEGCN.h5
    labels4DCVAEGCN.h5
    affectiveFeatures.h5
```

## Important environment note

This project was written for an older research stack. The included dependency versions are very old, so use a compatible Python version.

Recommended:

- Python 3.8
- Windows 10/11 using PowerShell
- CUDA-capable GPU is optional, but the scripts are written to use `cuda:0` by default

If you do not have a GPU, you may need to change the script to use CPU manually.

## 1) Create a virtual environment

Open PowerShell in the repository root and run:

```powershell
cd C:\Users\minh\Documents\STEP
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 2) Install project dependencies

```powershell
pip install -r requirements.txt
```

The repo also imports TensorFlow in some modules, but it is not listed in `requirements.txt`, so install it separately:

```powershell
pip install tensorflow-cpu==2.4.0
```

Then install the local utility package used by the project:

```powershell
pip install -e torchlight
```

## 3) Prepare the data folder

Create the top-level `data` directory at the repo root if it does not already exist:

```powershell
mkdir data
```

Then copy or download the dataset files into that folder. You need at least the real gait files and the label files for the experiments you want to run.

If you are using the Emotion-Gait dataset, place the downloaded `.h5` files so that all scripts can find them through relative paths like:

- `../data`
- `../../data`

## 4) Check the default run behavior

Each `main.py` script has command-line arguments and also defaults to training immediately. You should open the script before running to confirm the path and defaults.

For example, the classifier script in `classifier_stgcn_real_only/main.py` does:

```python
data_path = os.path.join(base_path, '../../data/')
...
if args.train:
    pr.train()
```

This means it will train automatically unless you modify the logic.

## 5) Run the real-only E-Gait Baseline-STEP reproduction

The real-only classifier defaults to `E-Gait/features_merged.h5` and
`E-Gait/labels_merged.h5`. It verifies the paired sample keys and emotion
labels, converts the documented 16-joint order, resamples each sequence to 75
frames, and uses a root-centered coordinate transform with one scale fitted on
the training split.

```powershell
cd C:\Users\minh\Documents\STEP\classifier_stgcn_real_only
..\.venv-legacy\Scripts\python.exe main.py
```

Every run first trains on a balanced 16-gait diagnostic set. The full 7:2:1
split and training start only if the tiny model reaches 100% training accuracy.
The default full run uses 100 epochs, batch size 8, Adam, learning rate 0.001,
weight decay 5e-4, and learning-rate drops at epochs 50, 75, and 88. These
optimizer settings match the prior stable real-only run. A
diagnostic-only run is available with:

```powershell
..\.venv-legacy\Scripts\python.exe main_quick_test.py
```

Run outputs, the tiny checkpoint, the full checkpoint, preprocessing metadata,
and the console log are stored under
`classifier_stgcn_real_only/model_classifier_stgcn/egait_2177_baseline/` and
`classifier_stgcn_real_only/logs/`.

## 6) Run the generator (CVAE)

The generator pipeline is in `generator_cvae/`.

```powershell
cd C:\Users\minh\Documents\STEP\generator_cvae
python main_stgcn.py
```

This script:

- loads the gait data from `../data`
- scales the data
- trains the CVAE ST-GCN model
- generates synthetic samples
- writes metadata into `model_gait_cvae_stgcn/info.txt`

If you only need a quick baseline run, you can start with the real-only classifier and come back to this later.

## 7) Run the controlled real + synthetic classifier

The controlled `real_and_synth` experiment uses the same 2,177 real gaits,
preprocessing, seed-42 stratified 7:2:1 split, and Baseline-STEP classifier as
the real-only run. It first requires the 16-sample overfit check to pass, then
fits the conditional STEP CVAE using only the 1,522 real training gaits. The
generated samples are added to classifier training; validation and test remain
the same held-out real gaits. This avoids using bundled synthetic files whose
generation split provenance is not recorded.

```powershell
cd C:\Users\minh\Documents\STEP\classifier_stgcn_real_and_synth
..\.venv-legacy\Scripts\python.exe main.py
```

Defaults generate 1,000 gaits per emotion and train the classifier for 100
epochs. Use `--cvae-epochs`, `--synth-per-class`, or `--num-epoch` to change
those settings. Each run writes a fixed real split manifest, CVAE checkpoint,
synthetic training archive, classifier checkpoint, metadata, and a log under
`classifier_stgcn_real_and_synth/`.

The completed local run (`20260924_183257`) passed the tiny-overfit check
(100% at epoch 15). On the identical real split, real-only scored 81.34%
validation / 80.54% test, while this real + synthetic run scored 76.04% /
75.11% (−5.30 / −5.43 percentage points). The synthetic samples had valid,
root-centered coordinates, but did not improve this classifier run. This is a
controlled experiment on the available 2,177-real-gait merge, not a numeric
reproduction of the paper's 4,227-real-gait result. The train-only CVAE uses a
mean-normalized version of the position/motion/KL objective; its exact loss is
recorded in each run's `cvae_metadata.json`.

## 8) Compute affective features

The hybrid pipeline depends on affective features extracted from the raw gait sequences.

Open `compute_aff_features/main.py` and check the hard-coded path near the top:

```python
base_path = '/mnt/q/Gamma/Gait'
```

Replace it with your local dataset directory, or create a matching folder structure. Then run:

```powershell
cd C:\Users\minh\Documents\STEP\compute_aff_features
python main.py
```

This script processes the feature data and saves outputs like `affectiveFeatures*.h5` under the data folder.

## 9) Run the hybrid classifier

Once the affective features are available, run:

```powershell
cd C:\Users\minh\Documents\STEP\classifier_hybrid
python main.py
```

This model combines:

- deep gait features
- physiologically-motivated affective features

and trains a hybrid classifier.

## 10) Common troubleshooting

### Error: missing dataset files

If you see a file-not-found or HDF5 error, make sure your `data/` folder contains the expected `.h5` files and that the script is running from the correct working directory.

### Error: CUDA not available

The code sets:

```python
device = 'cuda:0'
```

If your machine does not have CUDA, update it to:

```python
device = 'cpu'
```

at the top of the relevant script before running.

### Error: TensorFlow import fails

Install TensorFlow explicitly:

```powershell
pip install tensorflow-cpu==2.4.0
```

### Error: `torchlight` import fails

Install the repo package:

```powershell
pip install -e torchlight
```

### Error: package version conflicts

The repo is pinned to old versions such as:

- `numpy==1.19.0`
- `torch==1.5.1`
- `scikit-learn==0.23.1`
- `h5py==2.10.0`

This is why Python 3.8 is recommended. Using newer Python versions may create compatibility issues.

## Recommended run order for a clean reproduction

If you want to reproduce the full pipeline in order, do this:

```powershell
# 1. setup environment
cd C:\Users\minh\Documents\STEP
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install tensorflow-cpu==2.4.0
pip install -e torchlight

# 2. place dataset under data/

# 3. train baseline real-only model
cd classifier_stgcn_real_only
python main.py

# 4. fit the CVAE on the real-only training split and train real + synthetic
cd ..\classifier_stgcn_real_and_synth
..\.venv-legacy\Scripts\python.exe main.py

# 6. compute affective features
cd ..\compute_aff_features
python main.py

# 7. train hybrid model
cd ..\classifier_hybrid
python main.py
```

## Notes

- This repository is research code and not a polished end-user application.
- A good first step is to run the real-only classifier to verify the environment and the dataset are correct.
- The project expects a structured dataset and a compatible Python environment; the main challenge is usually dependency compatibility rather than the model code itself.

## Citation

If this project is useful for your work, please cite the paper as listed at the top of this file.

---

This README is intended to be your working guide for running the repository from a clean setup.
