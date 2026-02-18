# EEG-Based Cognitive Load Detection

An end-to-end pipeline for **detecting cognitive stress levels** from EEG brain signals collected during **Arithmetic** and **Stroop** tasks using Machine Learning and Deep Learning models.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Step-by-Step Instructions](#step-by-step-instructions)
7. [Models](#models)
8. [Results](#results)

---

## Project Overview

| Aspect | Detail |
|--------|--------|
| **Data Source** | Muse EEG headband (4 EEG + 2 AUX + 3 ACC channels) |
| **Tasks** | Arithmetic reasoning · Stroop color-word test |
| **Participants** | 15 per stress level per task |
| **Stress Levels** | Natural (0) · Low-Level (1) · Mid-Level (2) · High-Level (3) |
| **Features** | 77 statistical features (mean, std, min, max, median, skew, kurtosis) |
| **ML Models** | Random Forest · SVM · KNN · Gradient Boosting · Logistic Regression |
| **DL Models** | DNN · 1D-CNN · LSTM |

---

## Dataset Description

Raw EEG data is organized under `raw_data/` in two task folders:

- **`Arithmetic_Data/`** — EEG recordings during arithmetic reasoning tasks
- **`Stroop_Data/`** — EEG recordings during Stroop color-word tests

Each folder contains **60 text files** (4 stress levels × 15 participants):

```
<level>-<participant_id>.txt
```

where `<level>` is one of `natural`, `lowlevel`, `midlevel`, `highlevel` and `<participant_id>` ranges from 1 to 15.

Each `.txt` file contains 25 columns of raw sensor data including 4 EEG channels (TP9, AF7, AF8, TP10), 2 auxiliary channels, DRL, REF, 3 accelerometer axes, and metadata.

---

## Project Structure

```
├── app.py                  # Streamlit interactive dashboard
├── data_preprocessing.py   # Raw EEG → feature extraction pipeline
├── train_ml.ipynb          # ML model training notebook
├── train_dl.ipynb          # DL model training notebook
├── requirements.txt        # Python dependencies
├── processed_dataset.csv   # Preprocessed feature dataset (generated)
├── dl_results.json         # Deep Learning evaluation results
├── ml_results.json         # Machine Learning evaluation results (generated)
├── models/                 # Saved trained models
│   ├── dl_DNN.pt
│   ├── dl_CNN_1D.pt
│   ├── dl_LSTM.pt
│   ├── scaler.pkl
│   └── feature_cols.pkl
├── static/                 # Generated plots (confusion matrices, training history)
├── raw_data/               # Raw EEG recordings
│   ├── Arithmetic_Data/
│   └── Stroop_Data/
└── README.md
```

---

## Prerequisites

- **Python** 3.8 or higher
- **pip** package manager
- (Optional) A virtual environment tool (`venv`, `conda`)

---

## Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/EEG-Based-Cognitive-Load-Detection.git
cd EEG-Based-Cognitive-Load-Detection
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data manipulation |
| `scikit-learn` | ML models & evaluation |
| `torch` | Deep Learning models (DNN, CNN, LSTM) |
| `tensorflow` | Additional DL support |
| `streamlit` | Interactive web dashboard |
| `matplotlib`, `seaborn` | Visualization |
| `joblib` | Model serialization |

---

## Step-by-Step Instructions

### Step 1 — Data Preprocessing

Convert raw EEG text files into a structured feature dataset.

```bash
python data_preprocessing.py
```

**What it does:**
1. Reads all `.txt` files from `raw_data/Arithmetic_Data/` and `raw_data/Stroop_Data/`
2. For each recording, extracts **77 statistical features** (mean, std, min, max, median, skew, kurtosis) across 11 sensor channels
3. Labels each sample with the **task type**, **stress level**, and **participant ID**
4. Saves the result to **`processed_dataset.csv`**

**Output:**
```
✓ Saved processed dataset → processed_dataset.csv
  Shape: (120, 81)   # 120 samples × (77 features + 4 metadata columns)
```

---

### Step 2 — Train Machine Learning Models

Open and run the **`train_ml.ipynb`** notebook:

```bash
jupyter notebook train_ml.ipynb
```

**What it does:**
1. Loads `processed_dataset.csv`
2. Splits data into training and test sets
3. Scales features using `StandardScaler`
4. Trains 5 ML classifiers:
   - Random Forest
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Gradient Boosting
   - Logistic Regression
5. Evaluates each model (accuracy, precision, recall, F1-score, cross-validation)
6. Saves trained models to `models/ml_<ModelName>.pkl`
7. Saves evaluation metrics to `ml_results.json`
8. Generates confusion matrix plots in `static/`

---

### Step 3 — Train Deep Learning Models

Open and run the **`train_dl.ipynb`** notebook:

```bash
jupyter notebook train_dl.ipynb
```

**What it does:**
1. Loads `processed_dataset.csv`
2. Prepares PyTorch datasets and data loaders
3. Trains 3 DL architectures:
   - **DNN** — Fully connected deep neural network
   - **CNN_1D** — 1D Convolutional Neural Network
   - **LSTM** — Bidirectional LSTM network
4. Evaluates each model with cross-validation
5. Saves trained models to `models/dl_<ModelName>.pt`
6. Saves the scaler (`models/scaler.pkl`) and feature columns (`models/feature_cols.pkl`)
7. Saves evaluation metrics to `dl_results.json`
8. Generates confusion matrices and training history plots in `static/`

---

### Step 4 — Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser (default: `http://localhost:8501`) with 5 pages:

| Page | Description |
|------|-------------|
| **🏠 Overview** | Project summary, pipeline overview, and dataset statistics |
| **📊 Data Explorer** | Interactive feature distributions, correlation heatmaps, and boxplots |
| **🤖 ML Results** | ML model comparison charts, metrics table, and confusion matrices |
| **🧬 DL Results** | DL model comparison, training history curves, and confusion matrices |
| **🔮 Predict** | Upload a raw EEG file or select a sample to predict stress level using any trained model |

---

### Step 5 — Make Predictions

On the **🔮 Predict** page you can:

1. **Upload a raw EEG `.txt` file** — features are extracted automatically and passed to the selected model
2. **Use a dataset sample** — pick any sample from the processed dataset to verify predictions
3. **Manual input** — enter feature values directly

Choose between **Machine Learning** or **Deep Learning** models, select a specific model, and view:
- Predicted stress level (Natural / Low / Mid / High)
- Confidence score
- Class probability bar chart

---

## Models

### Machine Learning

| Model | File |
|-------|------|
| Random Forest | `models/ml_RandomForest.pkl` |
| SVM | `models/ml_SVM.pkl` |
| KNN | `models/ml_KNN.pkl` |
| Gradient Boosting | `models/ml_GradientBoosting.pkl` |
| Logistic Regression | `models/ml_LogisticRegression.pkl` |

### Deep Learning

| Model | Architecture | File |
|-------|-------------|------|
| DNN | 4-layer fully connected (256→128→64→4) | `models/dl_DNN.pt` |
| CNN_1D | 2-layer 1D convolution + FC head | `models/dl_CNN_1D.pt` |
| LSTM | 2-layer bidirectional LSTM + FC head | `models/dl_LSTM.pt` |

---

## Results

### Deep Learning Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | CV Accuracy |
|-------|----------|-----------|--------|----------|-------------|
| DNN | 0.375 | 0.249 | 0.375 | 0.285 | 0.400 ± 0.057 |
| CNN_1D | 0.375 | 0.343 | 0.375 | 0.348 | 0.442 ± 0.086 |
| LSTM | 0.333 | 0.274 | 0.333 | 0.282 | 0.375 ± 0.059 |

> ML results will be available after running `train_ml.ipynb`.

---

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess raw EEG data
python data_preprocessing.py

# 3. Train ML models (open in Jupyter and run all cells)
jupyter notebook train_ml.ipynb

# 4. Train DL models (open in Jupyter and run all cells)
jupyter notebook train_dl.ipynb

# 5. Launch the dashboard
streamlit run app.py
```
