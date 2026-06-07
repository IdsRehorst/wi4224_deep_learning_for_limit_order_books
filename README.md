# Deep Learning for Limit Order Books: Reduced Replication

Reduced replication of Sirignano's *Deep Learning for Limit Order Books* for the TU Delft course **WI4224 Special Topics in Financial Engineering**.

This project studies short-term prediction of joint best ask and best bid movements from limit order book data. Following the original paper, the main prediction problem is to estimate

\[
P((\Delta A_t, \Delta B_t) \mid X_t),
\]

where \(X_t\) is the current limit order book state, and \((\Delta A_t, \Delta B_t)\) is the future movement of the best ask and best bid prices over a fixed prediction horizon.

The final experiments use the **WSELOB-2017** dataset, a publicly available year-long limit order book dataset for the five largest companies listed on the Warsaw Stock Exchange. We reconstruct one-second order book snapshots and compare several probabilistic models:

- naive empirical baseline;
- multinomial logistic regression;
- standard feedforward neural network;
- reduced spatial neural network;
- full local spatial neural network.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── WSELOB-2017/
│   │       └── orders/
│   └── processed/
│
├── results/
│
├── figures/
│
├── src/
│   ├── convert_wselob.py
│   ├── train_empirical.py
│   ├── train_logistic.py
│   ├── train_mlp.py
│   ├── train_spatial.py
│   └── train_full_spatial.py
│
├── requirements.txt
└── README.md
```

The `data/raw/`, `data/processed/` and `results/` folders are not meant to be fully tracked by Git, since the raw and processed datasets are large.

---

## Data

The final experiments use **WSELOB-2017**:

```text
WSELOB-2017: The year-long database of limit order books
for the five biggest companies listed on the Warsaw Stock Exchange
```

The dataset can be downloaded from Mendeley Data:

```text
https://data.mendeley.com/datasets/3g4mhdp899/1
```

DOI:

```text
10.17632/3g4mhdp899.1
```

The raw dataset is not included in this repository because it is too large for GitHub. After downloading the dataset, place the order book files in:

```text
data/raw/WSELOB-2017/orders/
```

The expected files are HDF5 files such as:

```text
KGHM_lob_2017_zlib.h5
PEKAO_lob_2017_zlib.h5
PKNORLEN_lob_2017_zlib.h5
PKOBP_lob_2017_zlib.h5
PZU_lob_2017_zlib.h5
```

Only the order book files are needed for the experiments in this repository.

---

## Prediction Problem

For each reconstructed snapshot at time \(t\), we use the current order book state \(X_t\) to predict the future movement of the best ask and best bid over a fixed horizon \(\Delta t\).

The target is

\[
Y_t = (\Delta A_t, \Delta B_t),
\]

where both components are measured in tick units. Movements are clipped to a finite range:

```text
{-K, ..., -1, 0, 1, ..., K}^2
```

In the final experiments we use:

```text
K = 10
```

This gives:

```text
(2K + 1)^2 = 441
```

possible joint movement classes.

---

## Final Experimental Setup

The main experiments use the following setup:

```text
Dataset:              WSELOB-2017
Stocks:               KGHM, PEKAO, PKNORLEN, PKOBP, PZU
Sampling interval:    1 second
LOB levels used:      50
Prediction horizon:   30 seconds
Tick size:            5 raw price units
Clipping range:       [-10, 10] ticks
Number of classes:    441
Train split:          first 70%
Validation split:     next 15%
Test split:           final 15%
```

The split is chronological, so models are trained on earlier observations and evaluated on later observations.

---

## Quick Start

### 1. Convert a WSELOB stock file

Example for KGHM:

```bash
python src/convert_wselob.py \
    --input data/raw/WSELOB-2017/orders/KGHM_lob_2017_zlib.h5 \
    --output data/processed/WSELOB_KGHM_h30_L50_full_tick5 \
    --num-levels 50 \
    --sample-interval-seconds 1 \
    --horizon-seconds 30 \
    --clip-ticks 10 \
    --tick-size-raw 5 \
    --save-spatial-arrays \
    --save-dense-spatial-arrays \
    --dense-max-offset 50
```

This reconstructs order book snapshots, constructs features and labels, and saves the processed dataset.

The output folder contains arrays such as:

```text
X.npy
y_pair.npy
y_class.npy
feature_names.json
metadata.json
spatial/
```

The `spatial/` folder contains additional arrays needed for the full local spatial neural network.

---

### 2. Train the empirical baseline

```bash
python src/train_empirical.py \
    --processed-dir data/processed/WSELOB_KGHM_h30_L50_full_tick5 \
    --results-dir results/WSELOB_KGHM_h30_L50_full_tick5
```

The empirical baseline ignores the current order book state and estimates the unconditional distribution of the labels:

\[
\hat p(y \mid X_t) = \hat p(y).
\]

---

### 3. Train the logistic regression baseline

```bash
python src/train_logistic.py \
    --processed-dir data/processed/WSELOB_KGHM_h30_L50_full_tick5 \
    --results-dir results/WSELOB_KGHM_h30_L50_full_tick5
```

This trains a multinomial logistic regression model using mini-batch optimization. The mini-batch implementation is used because the full-year datasets contain around ten million samples per stock.

---

### 4. Train the standard neural network

```bash
python src/train_mlp.py \
    --processed-dir data/processed/WSELOB_KGHM_h30_L50_full_tick5 \
    --results-dir results/WSELOB_KGHM_h30_L50_full_tick5
```

This trains a feedforward neural network that maps the order book feature vector to a probability distribution over all 441 joint movement classes.

---

### 5. Train the reduced spatial neural network

```bash
python src/train_spatial.py \
    --processed-dir data/processed/WSELOB_KGHM_h30_L50_full_tick5 \
    --results-dir results/WSELOB_KGHM_h30_L50_full_tick5
```

The reduced spatial neural network uses a structured hazard-style output representation. It models movement magnitudes in an ordered way, while remaining computationally efficient.

---

### 6. Train the full local spatial neural network

```bash
python src/train_full_spatial.py \
    --processed-dir data/processed/WSELOB_KGHM_h30_L50_full_tick5 \
    --results-dir results/WSELOB_KGHM_h30_L50_full_tick5
```

The full local spatial neural network uses dense tick-grid arrays and local liquidity windows around candidate price movement levels. This model is closer to the spatial modelling idea from the original paper.

The script saves two checkpoints:

- best validation negative log-likelihood;
- best validation movement-conditional negative log-likelihood.

---

## Script Overview

### `convert_wselob.py`

Converts raw WSELOB-2017 HDF5 files into processed training datasets.

Main functionality:

- reads raw WSELOB order event files;
- reconstructs order book snapshots;
- samples snapshots at fixed one-second intervals;
- constructs flat order book features;
- computes future best ask and best bid movements;
- clips movements to a finite range;
- maps joint movements to class labels;
- saves metadata and processed arrays;
- optionally saves dense spatial arrays for the full local spatial network.

---

### `train_empirical.py`

Implements the naive empirical baseline.

Main functionality:

- estimates the unconditional class distribution on the training set;
- evaluates train, validation and test negative log-likelihood;
- computes movement-conditional negative log-likelihood;
- reports the most common predicted classes;
- saves results as JSON.

---

### `train_logistic.py`

Implements multinomial logistic regression.

Main functionality:

- standardizes features using the training set;
- trains a linear softmax classifier using mini-batch optimization;
- evaluates negative log-likelihood and movement-conditional negative log-likelihood;
- saves results as JSON.

---

### `train_mlp.py`

Implements the standard feedforward neural network.

Main functionality:

- standardizes features;
- trains a PyTorch MLP classifier;
- uses dropout and weight decay;
- applies early stopping based on validation negative log-likelihood;
- evaluates train, validation and test performance;
- saves metrics and the trained model.

---

### `train_spatial.py`

Implements the reduced spatial neural network.

Main functionality:

- uses the flat order book feature representation;
- models ask and bid movements using a structured hazard-style output;
- evaluates both full NLL and movement-conditional NLL;
- saves metrics and the trained model.

---

### `train_full_spatial.py`

Implements the full local spatial neural network.

Main functionality:

- loads dense spatial arrays from the processed dataset;
- extracts local liquidity windows around candidate tick levels;
- computes continuation probabilities using local spatial features;
- saves one checkpoint based on validation NLL;
- saves one checkpoint based on validation movement NLL;
- evaluates both checkpoints on the test set.

---

## Evaluation Metrics

The main evaluation metrics are:

- negative log-likelihood;
- movement-conditional negative log-likelihood;
- accuracy;
- macro-F1 score.

Negative log-likelihood is the primary metric because the goal is to estimate the full conditional distribution of future quote movements.

Movement-conditional negative log-likelihood is especially important because the data are highly imbalanced. The no-movement class \((0,0)\) is often dominant, so ordinary accuracy can be misleading.

---

## Notes

This repository does not include the raw WSELOB-2017 dataset or the processed datasets, because these files are too large for GitHub.

To reproduce the experiments:

1. download WSELOB-2017 from Mendeley Data;
2. place the HDF5 order files in `data/raw/WSELOB-2017/orders/`;
3. run `convert_wselob.py` for the desired stock;
4. run the training scripts on the processed dataset.

The code was developed for a reduced replication and course project. 
