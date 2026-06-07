from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def load_dataset(processed_dir: Path) -> tuple[np.ndarray, dict]:
    y_class = np.load(processed_dir / "y_class.npy", mmap_mode="r")

    with open(processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return y_class, metadata


def chronological_split(
    n_samples: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[slice, slice, slice]:
    n_train = int(train_fraction * n_samples)
    n_validation = int(validation_fraction * n_samples)

    train_slice = slice(0, n_train)
    validation_slice = slice(n_train, n_train + n_validation)
    test_slice = slice(n_train + n_validation, n_samples)

    return train_slice, validation_slice, test_slice


def decode_class(class_id: int, clip_ticks: int) -> tuple[int, int]:
    num_values = 2 * clip_ticks + 1

    delta_ask = class_id // num_values - clip_ticks
    delta_bid = class_id % num_values - clip_ticks

    return int(delta_ask), int(delta_bid)


def fit_empirical_distribution(
    y_train: np.ndarray,
    num_classes: int,
    smoothing: float = 1e-12,
) -> np.ndarray:
    y_train = np.asarray(y_train, dtype=np.int64)

    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)

    probabilities = (counts + smoothing) / (
        counts.sum() + smoothing * num_classes
    )

    return probabilities


def evaluate_empirical_model(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    zero_class: int,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)

    eps = 1e-15
    probabilities = np.clip(probabilities, eps, 1.0)

    sample_nll = -np.log(probabilities[y_true])
    nll = np.mean(sample_nll)

    movement_mask = y_true != zero_class

    if movement_mask.any():
        movement_nll = np.mean(sample_nll[movement_mask])
        movement_share = np.mean(movement_mask)
    else:
        movement_nll = np.nan
        movement_share = 0.0

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
        "movement_negative_log_likelihood": float(movement_nll),
        "movement_share": float(movement_share),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "predicted_class": predicted_class,
    }


def print_top_classes(
    probabilities: np.ndarray,
    clip_ticks: int,
    top_k: int = 10,
) -> None:
    order = np.argsort(probabilities)[::-1]

    print("Top empirical classes")
    print("---------------------")

    for class_id in order[:top_k]:
        delta_ask, delta_bid = decode_class(int(class_id), clip_ticks)
        probability = probabilities[class_id]

        print(
            f"class {class_id:>3} "
            f"({delta_ask:>3}, {delta_bid:>3}) : "
            f"{100 * probability:6.2f}%"
        )


def print_metrics(name: str, metrics: dict) -> None:
    print(
        f"{name:<11} "
        f"NLL: {metrics['negative_log_likelihood']:.4f} | "
        f"Move NLL: {metrics['movement_negative_log_likelihood']:.4f} | "
        f"Move share: {100 * metrics['movement_share']:.2f}% | "
        f"Accuracy: {100 * metrics['accuracy']:.2f}% | "
        f"Macro-F1: {metrics['macro_f1']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train naive empirical baseline."
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/WSELOB_PKNORLEN_h30_L50_full_tick5",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/WSELOB_PKNORLEN_h30_L50_full_tick5",
    )

    args = parser.parse_args()

    processed_dir = resolve_project_path(args.processed_dir)
    results_dir = resolve_project_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_class, metadata = load_dataset(processed_dir)

    num_samples = len(y_class)
    num_classes = int(metadata["num_classes"])
    clip_ticks = int(metadata["clip_ticks"])

    zero_class = clip_ticks * (2 * clip_ticks + 1) + clip_ticks

    train_slice, validation_slice, test_slice = chronological_split(num_samples)

    y_train = y_class[train_slice]
    y_validation = y_class[validation_slice]
    y_test = y_class[test_slice]

    probabilities = fit_empirical_distribution(
        y_train=y_train,
        num_classes=num_classes,
    )

    train_metrics = evaluate_empirical_model(
        probabilities=probabilities,
        y_true=y_train,
        zero_class=zero_class,
    )

    validation_metrics = evaluate_empirical_model(
        probabilities=probabilities,
        y_true=y_validation,
        zero_class=zero_class,
    )

    test_metrics = evaluate_empirical_model(
        probabilities=probabilities,
        y_true=y_test,
        zero_class=zero_class,
    )

    print("Naive empirical model")
    print("=====================")
    print(f"Processed directory:     {processed_dir}")
    print(f"Results directory:       {results_dir}")
    print(f"Number of samples:       {num_samples:,}")
    print(f"Training samples:        {len(y_train):,}")
    print(f"Validation samples:      {len(y_validation):,}")
    print(f"Test samples:            {len(y_test):,}")
    print(f"Number of classes:       {num_classes}")
    print(f"Zero class:              {zero_class}")
    print()

    print_top_classes(probabilities, clip_ticks=clip_ticks)
    print()

    print("Metrics")
    print("-------")
    print_metrics("Train", train_metrics)
    print_metrics("Validation", validation_metrics)
    print_metrics("Test", test_metrics)

    output = {
        "model": "naive_empirical",
        "metadata": metadata,
        "zero_class": int(zero_class),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "probabilities": probabilities.tolist(),
    }

    output_path = results_dir / "empirical_baseline.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()