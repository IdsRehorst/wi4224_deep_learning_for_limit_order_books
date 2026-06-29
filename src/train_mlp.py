from pathlib import Path
import argparse
import json
import time
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    X = np.load(processed_dir / "X.npy", mmap_mode="r")
    y_class = np.load(processed_dir / "y_class.npy", mmap_mode="r")

    with open(processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return X, y_class, metadata


def chronological_split(
    n_samples: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[int, int, int]:
    n_train = int(train_fraction * n_samples)
    n_validation = int(validation_fraction * n_samples)
    n_test = n_samples - n_train - n_validation

    return n_train, n_validation, n_test


def decode_class(class_id: int, clip_ticks: int) -> tuple[int, int]:
    num_values = 2 * clip_ticks + 1

    delta_ask = class_id // num_values - clip_ticks
    delta_bid = class_id % num_values - clip_ticks

    return int(delta_ask), int(delta_bid)


def iter_sequential_batches(start: int, end: int, batch_size: int):
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        yield slice(batch_start, batch_end)


def iter_shuffled_batches(
    start: int,
    end: int,
    batch_size: int,
    rng: np.random.Generator,
):
    indices = np.arange(start, end)
    rng.shuffle(indices)

    for batch_start in range(0, len(indices), batch_size):
        batch_end = min(batch_start + batch_size, len(indices))
        yield indices[batch_start:batch_end]


def fit_scaler_in_batches(
    X: np.ndarray,
    train_end: int,
    batch_size: int,
) -> StandardScaler:
    scaler = StandardScaler()

    for batch_slice in iter_sequential_batches(0, train_end, batch_size):
        X_batch = np.asarray(X[batch_slice], dtype=np.float32)
        scaler.partial_fit(X_batch)

    return scaler


def transform_batch(
    X_batch: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    X_batch = np.asarray(X_batch, dtype=np.float32)

    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale = np.where(scale == 0.0, 1.0, scale)

    return ((X_batch - mean) / scale).astype(np.float32)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def evaluate_nll_metrics(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
    zero_class: int,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    movement_loss = 0.0
    movement_samples = 0

    with torch.no_grad():
        for batch_slice in iter_sequential_batches(start, end, batch_size):
            X_batch_np = transform_batch(X[batch_slice], scaler)
            y_batch_np = np.asarray(y[batch_slice], dtype=np.int64)

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            logits = model(X_batch)
            losses = F.cross_entropy(logits, y_batch, reduction="none")

            total_loss += float(losses.sum().item())
            total_samples += int(y_batch.shape[0])

            movement_mask = y_batch != zero_class

            if movement_mask.any():
                movement_loss += float(losses[movement_mask].sum().item())
                movement_samples += int(movement_mask.sum().item())

    nll = total_loss / total_samples
    movement_nll = movement_loss / movement_samples if movement_samples > 0 else np.nan
    movement_share = movement_samples / total_samples

    return {
        "negative_log_likelihood": float(nll),
        "movement_negative_log_likelihood": float(movement_nll),
        "movement_share": float(movement_share),
    }


def evaluate_full(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
    num_classes: int,
    zero_class: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    movement_loss = 0.0
    movement_samples = 0

    all_y_true = []
    all_y_pred = []

    movement_prob_sum = np.zeros(num_classes, dtype=np.float64)
    movement_prob_samples = 0

    with torch.no_grad():
        for batch_slice in iter_sequential_batches(start, end, batch_size):
            X_batch_np = transform_batch(X[batch_slice], scaler)
            y_batch_np = np.asarray(y[batch_slice], dtype=np.int64)

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            logits = model(X_batch)
            losses = F.cross_entropy(logits, y_batch, reduction="none")

            y_pred = torch.argmax(logits, dim=1)

            total_loss += float(losses.sum().item())
            total_samples += int(y_batch.shape[0])

            movement_mask = y_batch != zero_class

            if movement_mask.any():
                movement_loss += float(losses[movement_mask].sum().item())
                movement_count = int(movement_mask.sum().item())
                movement_samples += movement_count

                # We only softmax the movement subset to avoid storing full test predictions.
                movement_logits = logits[movement_mask]
                movement_probs = F.softmax(movement_logits, dim=1)

                movement_prob_sum += movement_probs.sum(dim=0).cpu().numpy()
                movement_prob_samples += movement_count

            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    nll = total_loss / total_samples
    movement_nll = movement_loss / movement_samples if movement_samples > 0 else np.nan
    movement_share = movement_samples / total_samples

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    )

    if movement_prob_samples > 0:
        avg_probs_movement = movement_prob_sum / movement_prob_samples
    else:
        avg_probs_movement = np.full(num_classes, np.nan)

    metrics = {
        "negative_log_likelihood": float(nll),
        "movement_negative_log_likelihood": float(movement_nll),
        "movement_share": float(movement_share),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "movement_probability_samples": int(movement_prob_samples),
    }

    return metrics, y_pred, avg_probs_movement


def print_most_common_predictions(
    y_pred: np.ndarray,
    clip_ticks: int,
    top_k: int = 10,
) -> None:
    unique, counts = np.unique(y_pred, return_counts=True)
    order = np.argsort(counts)[::-1]

    print("Most common predicted classes on test set")
    print("-----------------------------------------")

    for idx in order[:top_k]:
        class_id = int(unique[idx])
        count = int(counts[idx])
        delta_ask, delta_bid = decode_class(class_id, clip_ticks)

        print(
            f"class {class_id:>3} "
            f"({delta_ask:>3}, {delta_bid:>3}) : "
            f"{count:>8,} ({100 * count / len(y_pred):6.2f}%)"
        )


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    train_end: int,
    validation_start: int,
    validation_end: int,
    batch_size: int,
    eval_batch_size: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    zero_class: int,
) -> tuple[nn.Module, list[dict], float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_nll = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    history = []
    start_time = time.time()

    rng = np.random.default_rng(seed)

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        running_samples = 0

        for batch_indices in iter_shuffled_batches(
            start=0,
            end=train_end,
            batch_size=batch_size,
            rng=rng,
        ):
            X_batch_np = transform_batch(X[batch_indices], scaler)
            y_batch_np = np.asarray(y[batch_indices], dtype=np.int64)

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            loss.backward()
            optimizer.step()

            current_batch_size = int(y_batch.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            running_samples += current_batch_size

        train_epoch_nll = running_loss / running_samples

        validation_metrics = evaluate_nll_metrics(
            model=model,
            X=X,
            y=y,
            scaler=scaler,
            start=validation_start,
            end=validation_end,
            batch_size=eval_batch_size,
            device=device,
            zero_class=zero_class,
        )

        validation_nll = validation_metrics["negative_log_likelihood"]

        epoch_result = {
            "epoch": epoch,
            "train_epoch_nll": float(train_epoch_nll),
            "validation_nll": float(validation_nll),
            "validation_movement_nll": float(
                validation_metrics["movement_negative_log_likelihood"]
            ),
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:>3} | "
            f"train NLL={train_epoch_nll:.4f} | "
            f"val NLL={validation_nll:.4f} | "
            f"val move NLL={validation_metrics['movement_negative_log_likelihood']:.4f}"
        )

        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print()
            print(f"Early stopping after {epoch} epochs.")
            break

    training_time = time.time() - start_time

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history, training_time


def print_metrics(name: str, metrics: dict) -> None:
    if "accuracy" in metrics:
        print(
            f"{name:<11} "
            f"NLL: {metrics['negative_log_likelihood']:.4f} | "
            f"Move NLL: {metrics['movement_negative_log_likelihood']:.4f} | "
            f"Move share: {100 * metrics['movement_share']:.2f}% | "
            f"Accuracy: {100 * metrics['accuracy']:.2f}% | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )
    else:
        print(
            f"{name:<11} "
            f"NLL: {metrics['negative_log_likelihood']:.4f} | "
            f"Move NLL: {metrics['movement_negative_log_likelihood']:.4f} | "
            f"Move share: {100 * metrics['movement_share']:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train standard MLP baseline."
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/WSELOB_KGHM_h30_L50_full_tick5",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/WSELOB_KGHM_h30_L50_full_tick5",
    )

    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    processed_dir = resolve_project_path(args.processed_dir)
    results_dir = resolve_project_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y_class, metadata = load_dataset(processed_dir)

    num_samples = len(y_class)
    input_dim = int(X.shape[1])
    num_classes = int(metadata["num_classes"])
    clip_ticks = int(metadata["clip_ticks"])

    zero_class = clip_ticks * (2 * clip_ticks + 1) + clip_ticks

    n_train, n_validation, n_test = chronological_split(num_samples)

    train_start = 0
    train_end = n_train

    validation_start = n_train
    validation_end = n_train + n_validation

    test_start = validation_end
    test_end = num_samples

    device = get_device()

    print("Standard neural network baseline")
    print("================================")
    print(f"Device:                  {device}")
    print(f"Processed directory:     {processed_dir}")
    print(f"Results directory:       {results_dir}")
    print(f"Number of samples:       {num_samples:,}")
    print(f"Training samples:        {n_train:,}")
    print(f"Validation samples:      {n_validation:,}")
    print(f"Test samples:            {n_test:,}")
    print(f"Number of features:      {input_dim}")
    print(f"Number of classes:       {num_classes}")
    print(f"Zero class:              {zero_class}")
    print(f"Hidden layers:           {args.hidden_dims}")
    print(f"Dropout:                 {args.dropout}")
    print(f"Learning rate:           {args.learning_rate}")
    print(f"Weight decay:            {args.weight_decay}")
    print()

    print("Fitting scaler in batches...")
    scaler = fit_scaler_in_batches(
        X=X,
        train_end=train_end,
        batch_size=args.eval_batch_size,
    )
    print("Scaler fitted.")
    print()

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    model, history, training_time = train_model(
        model=model,
        X=X,
        y=y_class,
        scaler=scaler,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        zero_class=zero_class,
    )

    print()
    print(f"Training time:           {training_time:.2f} seconds")
    print()

    train_metrics = evaluate_nll_metrics(
        model=model,
        X=X,
        y=y_class,
        scaler=scaler,
        start=train_start,
        end=train_end,
        batch_size=args.eval_batch_size,
        device=device,
        zero_class=zero_class,
    )

    validation_metrics = evaluate_nll_metrics(
        model=model,
        X=X,
        y=y_class,
        scaler=scaler,
        start=validation_start,
        end=validation_end,
        batch_size=args.eval_batch_size,
        device=device,
        zero_class=zero_class,
    )

    test_metrics, test_predictions, test_avg_probs_movement = evaluate_full(
        model=model,
        X=X,
        y=y_class,
        scaler=scaler,
        start=test_start,
        end=test_end,
        batch_size=args.eval_batch_size,
        device=device,
        num_classes=num_classes,
        zero_class=zero_class,
    )

    print_most_common_predictions(
        y_pred=test_predictions,
        clip_ticks=clip_ticks,
    )

    print()
    print("Final metrics for best validation model")
    print("---------------------------------------")
    print_metrics("Train", train_metrics)
    print_metrics("Validation", validation_metrics)
    print_metrics("Test", test_metrics)

    output = {
        "model": "standard_mlp",
        "metadata": metadata,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "clip_ticks": clip_ticks,
        "zero_class": int(zero_class),
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "training_time_seconds": float(training_time),
        "history": history,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    output_path = results_dir / "standard_mlp.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    avg_probs_path = results_dir / "standard_mlp_avg_probs_movement_test.npy"

    np.save(
        avg_probs_path,
        test_avg_probs_movement,
    )

    model_path = results_dir / "standard_mlp.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "metadata": metadata,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        model_path,
    )

    print()
    print(f"Saved results to: {output_path}")
    print(f"Saved movement average probabilities to: {avg_probs_path}")
    print(f"Saved model to:   {model_path}")


if __name__ == "__main__":
    main()