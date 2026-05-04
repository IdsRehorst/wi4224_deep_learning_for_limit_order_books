# Deep Learning for Limit Order Books: Reduced Replication

Reduced replication of Sirignano's *Deep Learning for Limit Order Books* for the TU Delft course **WI4224 Special Topics in Financial Engineering**.

The project studies short-term prediction of joint best ask and best bid movements from limit order book data. We use LOBSTER sample data to build a smaller-scale version of the original problem and compare several models:

- naive empirical baseline;
- logistic regression baseline;
- standard neural network baseline.

The main prediction problem is to estimate

\[
P((\Delta A_t, \Delta B_t) \mid X_t),
\]

where \(X_t\) is the current limit order book state, and \((\Delta A_t, \Delta B_t)\) is the future movement of the best ask and best bid prices over a fixed horizon.

---

## Project Structure

~~~text
.
├── data/
│   ├── raw/
│   │   └── LOBSTER_SampleFile_AAPL_2012-06-21_50/
│   └── processed/
│
├── results/
│
├── src/
│   ├── load_lobster.py
│   ├── make_dataset.py
│   ├── train_empirical.py
│   ├── train_logistic.py
│   └── train_mlp.py
│
└── README.md
~~~

---

## Data

This project currently uses the LOBSTER sample file:

~~~text
LOBSTER_SampleFile_AAPL_2012-06-21_50
~~~

Place this folder inside:

~~~text
data/raw/
~~~

The folder should contain a message file and an orderbook file, for example:

~~~text
AAPL_2012-06-21_34200000_37800000_message_50.csv
AAPL_2012-06-21_34200000_37800000_orderbook_50.csv
~~~

LOBSTER prices are stored as integers multiplied by `10000`. Therefore, for U.S. stocks, a one-cent tick corresponds to `100` in the raw data.

---

## Quick Start

### 1. Check that the raw LOBSTER files can be loaded

~~~bash
python src/load_lobster.py
~~~

This script loads the message and orderbook files, combines them into a single dataframe, and prints basic sanity checks such as:

- number of rows;
- number of order book levels;
- first and last timestamp;
- best ask and best bid ranges;
- spread statistics.

### 2. Construct the processed dataset

~~~bash
python src/make_dataset.py
~~~

This script constructs the supervised learning dataset. It creates:

~~~text
data/processed/AAPL_2012-06-21_50/
├── X.npy
├── y_pair.npy
├── y_class.npy
├── feature_names.json
└── metadata.json
~~~

The feature matrix `X.npy` contains order book features such as transformed ask sizes, bid sizes and imbalance features.

The label files contain the future joint movement

~~~text
(Delta ask, Delta bid)
~~~

over a fixed prediction horizon.

The labels are clipped to a finite range:

~~~text
{-K, ..., -1, 0, 1, ..., K}^2
~~~

so that the problem can be treated as multiclass classification.

### 3. Train the naive empirical baseline

~~~bash
python src/train_empirical.py
~~~

This model estimates the unconditional empirical distribution of the labels:

\[
\hat p(y \mid X_t) = \hat p(y).
\]

It ignores the current order book state and serves as the simplest benchmark.

### 4. Train the logistic regression baseline

~~~bash
python src/train_logistic.py
~~~

This trains a multiclass logistic regression model using the processed order book features. The script performs a small regularization sweep over different values of `C` and selects the best model based on validation negative log-likelihood.

### 5. Train the standard neural network baseline

~~~bash
python src/train_mlp.py
~~~

This trains a feedforward neural network that maps the order book feature vector to a probability distribution over all joint price movement classes.

The model is trained using cross-entropy loss and evaluated using:

- negative log-likelihood;
- accuracy;
- macro-F1 score.

---

## Script Overview

### `load_lobster.py`

Loads raw LOBSTER message and orderbook files.

Main functionality:

- reads the message file;
- reads the orderbook file;
- assigns clear column names;
- combines both files;
- extracts best ask, best bid and spread;
- prints a short data summary.

### `make_dataset.py`

Creates the supervised learning dataset.

Main functionality:

- constructs feature vectors \(X_t\);
- computes future best ask and best bid movements;
- clips movements to a finite range;
- maps joint movements to class labels;
- saves processed arrays and metadata.

### `train_empirical.py`

Implements the naive empirical baseline.

Main functionality:

- estimates the unconditional class distribution on the training set;
- evaluates train, validation and test negative log-likelihood;
- reports the most common empirical classes.

### `train_logistic.py`

Implements the logistic regression baseline.

Main functionality:

- standardizes features;
- trains multiclass logistic regression;
- performs a regularization sweep;
- selects the best model based on validation negative log-likelihood;
- evaluates the selected model on the test set.

### `train_mlp.py`

Implements the standard feedforward neural network baseline.

Main functionality:

- standardizes features;
- trains an MLP classifier using PyTorch;
- uses early stopping based on validation negative log-likelihood;
- saves metrics and the trained model.

---

## Current Experimental Setup

The current default setup is:

~~~text
Ticker:               AAPL
Date:                 2012-06-21
LOB levels used:       10
Prediction horizon:    1 second
Clipping range:        [-10, 10] ticks
Number of classes:     441
~~~

These values can be changed in `src/make_dataset.py`.

---

## Notes

This is a reduced replication. The original paper uses a much larger NASDAQ Level III dataset and trains models across hundreds of stocks. Here, we first focus on building a correct and reproducible pipeline on the LOBSTER sample data. The same pipeline can later be applied to larger datasets if more data becomes available.
