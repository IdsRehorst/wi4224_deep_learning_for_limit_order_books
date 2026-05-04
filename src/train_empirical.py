from pathlib import Path
import json

import numpy as np
from sklearn.metrics import f1_score


def get_project_root() -> Path:
    """
    Assumes this file is located in project/src/.
    """
    return Path(__file__).resolve().parents[1]


def load_dataset(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load processed labels and metadata.
    """
    y_class = np.load(processed_dir / "y_class.npy")
    y_pair = np.load(processed_dir / "y_pair.npy")

    with open(processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return y_class, y_pair, metadata


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


def fit_empirical_distribution(
    y_train: np.ndarray,
    num_classes: int,
    smoothing: float = 1e-12,
) -> np.ndarray:
    """
    Estimate the unconditional empirical class distribution.

    A tiny smoothing value is used to avoid log(0) on validation/test data.
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)

    probabilities = (counts + smoothing) / (
        counts.sum() + smoothing * num_classes
    )

    return probabilities


def evaluate_empirical_model(
    probabilities: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    Evaluate the empirical distribution model.

    The probabilistic prediction is the same for every sample.
    The class prediction is the mode of the training distribution.
    """
    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1.0)

    nll = -np.mean(np.log(probabilities[y_true]))

    predicted_class = int(np.argmax(probabilities))
    y_pred = np.full_like(y_true, fill_value=predicted_class)

    accuracy = np.mean(y_pred == y_true)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(len(probabilities)),
        average="macro",
        zero_division=0,
    )

    return {
        "negative_log_likelihood": float(nll),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "predicted_class": predicted_class,
    }


def decode_class(class_id: int, clip_ticks: int) -> tuple[int, int]:
    """
    Convert a joint class index back to (Delta A, Delta B).
    """
    num_values = 2 * clip_ticks + 1

    delta_ask = class_id // num_values - clip_ticks
    delta_bid = class_id % num_values - clip_ticks

    return int(delta_ask), int(delta_bid)


def print_top_classes(
    probabilities: np.ndarray,
    clip_ticks: int,
    top_k: int = 10,
) -> None:
    """
    Print the most likely classes under the empirical distribution.
    """
    order = np.argsort(probabilities)[::-1]

    print("Top empirical classes")
    print("---------------------")

    for class_id in order[:top_k]:
        delta_ask, delta_bid = decode_class(class_id, clip_ticks)
        probability = probabilities[class_id]

        print(
            f"class {class_id:>2} "
            f"({delta_ask:>2}, {delta_bid:>2}) : "
            f"{100 * probability:6.2f}%"
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

    y_class, y_pair, metadata = load_dataset(processed_dir)

    num_samples = len(y_class)
    num_classes = int(metadata["num_classes"])
    clip_ticks = int(metadata["clip_ticks"])

    train_idx, validation_idx, test_idx = chronological_split(num_samples)

    y_train = y_class[train_idx]
    y_validation = y_class[validation_idx]
    y_test = y_class[test_idx]

    probabilities = fit_empirical_distribution(
        y_train=y_train,
        num_classes=num_classes,
    )

    train_metrics = evaluate_empirical_model(probabilities, y_train)
    validation_metrics = evaluate_empirical_model(probabilities, y_validation)
    test_metrics = evaluate_empirical_model(probabilities, y_test)

    print("Naive empirical model")
    print("=====================")
    print(f"Number of samples:       {num_samples:,}")
    print(f"Training samples:        {len(train_idx):,}")
    print(f"Validation samples:      {len(validation_idx):,}")
    print(f"Test samples:            {len(test_idx):,}")
    print(f"Number of classes:       {num_classes}")
    print()

    print_top_classes(probabilities, clip_ticks=clip_ticks)
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
        "model": "naive_empirical",
        "metadata": metadata,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "probabilities": probabilities.tolist(),
    }

    with open(results_dir / "empirical_baseline.json", "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved results to: {results_dir / 'empirical_baseline.json'}")


if __name__ == "__main__":
    main()