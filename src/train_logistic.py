from pathlib import Path
import json
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """
    Assumes this file is located in project/src/.
    """
    return Path(__file__).resolve().parents[1]


def load_dataset(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load processed feature matrix, class labels and metadata.
    """
    X = np.load(processed_dir / "X.npy")
    y_class = np.load(processed_dir / "y_class.npy")

    with open(processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return X, y_class, metadata


def chronological_split(
    n_samples: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological train/validation/test split.

    Since this is time-series data, we avoid random shuffling.
    """
    n_train = int(train_fraction * n_samples)
    n_validation = int(validation_fraction * n_samples)

    train_idx = np.arange(0, n_train)
    validation_idx = np.arange(n_train, n_train + n_validation)
    test_idx = np.arange(n_train + n_validation, n_samples)

    return train_idx, validation_idx, test_idx


def decode_class(class_id: int, clip_ticks: int) -> tuple[int, int]:
    """
    Convert class index back to (Delta A, Delta B).
    """
    num_values = 2 * clip_ticks + 1

    delta_ask = class_id // num_values - clip_ticks
    delta_bid = class_id % num_values - clip_ticks

    return int(delta_ask), int(delta_bid)


def full_probability_matrix(
    clf: LogisticRegression,
    X: np.ndarray,
    num_classes: int,
    eps: float = 1e-15,
) -> np.ndarray:
    """
    sklearn only returns probabilities for classes seen during training.
    This function expands them to all possible classes.

    Classes not observed in the training set receive a tiny probability.
    """
    proba_seen = clf.predict_proba(X)
    n_samples = X.shape[0]

    proba_full = np.full(
        shape=(n_samples, num_classes),
        fill_value=eps,
        dtype=np.float64,
    )

    seen_classes = clf.classes_.astype(int)

    # Reserve a tiny amount of mass for unseen classes.
    missing_mass = eps * (num_classes - len(seen_classes))
    proba_full[:, seen_classes] = proba_seen * (1.0 - missing_mass)

    # Numerical safety.
    proba_full = proba_full / proba_full.sum(axis=1, keepdims=True)

    return proba_full


def evaluate_probabilistic_classifier(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Evaluate probabilistic multi-class predictions.
    """
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1.0)

    y_pred = np.argmax(probabilities, axis=1)

    nll = -np.mean(np.log(probabilities[np.arange(len(y_true)), y_true]))
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(probabilities.shape[1]),
        average="macro",
        zero_division=0,
    )

    return {
        "negative_log_likelihood": float(nll),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def print_most_common_predictions(
    probabilities: np.ndarray,
    clip_ticks: int,
    top_k: int = 10,
) -> None:
    """
    Print the most commonly predicted classes.
    """
    y_pred = np.argmax(probabilities, axis=1)
    unique, counts = np.unique(y_pred, return_counts=True)

    order = np.argsort(counts)[::-1]

    print("Most common predicted classes")
    print("-----------------------------")

    for idx in order[:top_k]:
        class_id = int(unique[idx])
        count = int(counts[idx])
        delta_ask, delta_bid = decode_class(class_id, clip_ticks)

        print(
            f"class {class_id:>3} "
            f"({delta_ask:>3}, {delta_bid:>3}) : "
            f"{count:>8,} ({100 * count / len(y_pred):6.2f}%)"
        )


def main() -> None:
    project_root = get_project_root()

    processed_dir = (
        project_root
        / "data"
        / "processed"
        / "AAPL_2012-06-21_50"
    )

    results_dir = (
        project_root
        / "results"
        / "AAPL_2012-06-21_50"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y_class, metadata = load_dataset(processed_dir)

    num_samples = len(y_class)
    num_classes = int(metadata["num_classes"])
    clip_ticks = int(metadata["clip_ticks"])

    train_idx, validation_idx, test_idx = chronological_split(num_samples)

    X_train = X[train_idx]
    X_validation = X[validation_idx]
    X_test = X[test_idx]

    y_train = y_class[train_idx]
    y_validation = y_class[validation_idx]
    y_test = y_class[test_idx]

    print("Logistic regression baseline")
    print("============================")
    print(f"Number of samples:       {num_samples:,}")
    print(f"Training samples:        {len(train_idx):,}")
    print(f"Validation samples:      {len(validation_idx):,}")
    print(f"Test samples:            {len(test_idx):,}")
    print(f"Number of features:      {X.shape[1]}")
    print(f"Number of classes:       {num_classes}")
    print()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        verbose=1,
    )

    start_time = time.time()

    clf.fit(X_train_scaled, y_train)

    training_time = time.time() - start_time

    print()
    print(f"Training time:           {training_time:.2f} seconds")
    print(f"Classes seen in train:   {len(clf.classes_)} / {num_classes}")
    print()

    train_probabilities = full_probability_matrix(
        clf=clf,
        X=X_train_scaled,
        num_classes=num_classes,
    )
    validation_probabilities = full_probability_matrix(
        clf=clf,
        X=X_validation_scaled,
        num_classes=num_classes,
    )
    test_probabilities = full_probability_matrix(
        clf=clf,
        X=X_test_scaled,
        num_classes=num_classes,
    )

    train_metrics = evaluate_probabilistic_classifier(
        probabilities=train_probabilities,
        y_true=y_train,
    )
    validation_metrics = evaluate_probabilistic_classifier(
        probabilities=validation_probabilities,
        y_true=y_validation,
    )
    test_metrics = evaluate_probabilistic_classifier(
        probabilities=test_probabilities,
        y_true=y_test,
    )

    print_most_common_predictions(
        probabilities=test_probabilities,
        clip_ticks=clip_ticks,
    )

    print()
    print("Metrics")
    print("-------")
    print(
        f"Train NLL:       {train_metrics['negative_log_likelihood']:.4f} | "
        f"Accuracy: {100 * train_metrics['accuracy']:.2f}% | "
        f"Macro-F1: {train_metrics['macro_f1']:.4f}"
    )
    print(
        f"Validation NLL:  {validation_metrics['negative_log_likelihood']:.4f} | "
        f"Accuracy: {100 * validation_metrics['accuracy']:.2f}% | "
        f"Macro-F1: {validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test NLL:        {test_metrics['negative_log_likelihood']:.4f} | "
        f"Accuracy: {100 * test_metrics['accuracy']:.2f}% | "
        f"Macro-F1: {test_metrics['macro_f1']:.4f}"
    )

    output = {
        "model": "logistic_regression",
        "metadata": metadata,
        "training_time_seconds": training_time,
        "num_classes_seen_in_train": int(len(clf.classes_)),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "regularization_C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    }

    with open(results_dir / "logistic_regression.json", "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved results to: {results_dir / 'logistic_regression.json'}")


if __name__ == "__main__":
    main()