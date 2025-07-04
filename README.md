# 🧠 Seizure Detection and Prediction
EEG-based Seizure Detection using Random Forest Pipelines
Kaggle Challenge – Seizure Detection

## 📄 Overview
This project implements a pipeline to detect and predict epileptic seizures using EEG data. Leveraging Random Forest classifiers, we process .mat files representing brain signals from dogs and human patients, aiming to distinguish between ictal (seizure), interictal (non-seizure), and test phases.

🏆 Based on data from a Kaggle competition, this work was also published as a research paper exploring machine learning techniques for biomedical time-series classification.

## 🛠️ Environment & Dependencies
Python: 2.7

Virtual Environment: virtualenv

Dependencies:

hickle==2.1.0

numpy==1.14.0

scikit-learn==0.19.1

scipy==1.0.0

## 🔧 Installation
```
-- Set up virtual environment
virtualenv venv
source venv/bin/activate
```

Upgrade pip and install requirements
```
pip install --upgrade pip
pip install -r requirements.txt
```

## 📁 Project Structure
```
seizure-detection/
│
├── seizure-data/             # EEG data directory (see below)
│   ├── Dog_1/
│   ├── Dog_2/
│   └── ...
│
├── data-cache/               # Auto-generated: stores cached classifier and task results
│
├── submissions/              # Auto-generated: stores prediction submission files
│
├── SETTINGS.json             # Config file for paths
├── train.py                  # Module to train Random Forest models
├── predict.py                # Module to make predictions
├── cross_validation.py       # Module for cross-validation evaluation
└── seizure/                  # Core task logic

```
## 📦 Data Preparation

Place the EEG .mat files as shown below. The folder structure must match the entries in SETTINGS.json.
```
seizure-data/
  Dog_1/
    Dog_1_ictal_segment_1.mat
    Dog_1_interictal_segment_1.mat
    Dog_1_test_segment_1.mat
    ...
  Dog_2/
  ...
```
🧠 SETTINGS.json Sample
```
{
  "competition-data-dir": "seizure-data",
  "data-cache-dir": "data-cache",
  "submission-dir": "./submissions"
}
```
## 🚀 Training the Model
Trains one classifier per patient (dog or human) using EEG features.
```
source venv/bin/activate
python -m train
```
Models are saved in the data-cache/ directory as .pickle files.

Example:
data-cache/classifier_Dog_1_fft-with-time-freq-corr-1-48-r400-usf-gen1_rf3000mss1Bfrs0.pickle

## 🔮 Making Predictions
```
python -m predict
```
Uses trained classifiers to predict labels on test segments.

Predictions are saved in ./submissions/ (or wherever defined in SETTINGS.json).

⚠️ Takes ~2 hours to run depending on the dataset size.

To speed it up: edit the n_jobs parameter in seizure_detection.py to enable parallel processing.

💡 You can swap out test segments to run predictions on a new dataset — just ensure files are sequentially numbered from 1 onward:

```
Dog_1_test_segment_1.mat
Dog_1_test_segment_2.mat
```

## 🔁 Cross-Validation
Performs leave-one-seizure-out validation:
```
python -m cross_validation
```
Each fold trains on n–1 seizures and tests on the held-out one.

Gives a robust estimation of classifier generalization on unseen seizure events.



## Authors

- [@MuktaPatil](https://github.com/MuktaPatil)
- [@UdayanGaikwad](https://github.com/ggwpfax)

