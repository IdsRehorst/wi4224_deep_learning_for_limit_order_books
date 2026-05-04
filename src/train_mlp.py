from pathlib import Path
import json
import time
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def get_project_root() -> Path:
    """
    Assumes this file is located in project/src/.
    """
    return Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Use CUDA if available, otherwise Apple MPS if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_dataset(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
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
    n_train = int(train_fraction * n_samples)
    n_validation = int(validation_fraction * n_samples)

    train_idx = np.arange(0, n_train)
    validation_idx = np.arange(n_train, n_train + n_validation)
    test_idx = np.arange(n_train + n_validation, n_samples)

    return train_idx, validation_idx, test_idx


def decode_class(class_id: int, clip_ticks: int) -> tuple[int, int]:
    num_values = 2 * clip_ticks + 1

    delta_ask = class_id // num_values - clip_ticks
    delta_bid = class_id % num_values - clip_ticks

    return int(delta_ask), int(delta_bid)


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


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_y_true = []
    all_y_pred = []

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += float(loss.item())
            total_samples += int(y_batch.shape[0])

            y_pred = torch.argmax(logits, dim=1)

            all_y_true.append(y_batch.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    nll = total_loss / total_samples
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        average="macro",
        zero_division=0,
    )

    return {
        "negative_log_likelihood": float(nll),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    predictions = []

    with torch.no_grad():
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            y_pred = torch.argmax(logits, dim=1)

            predictions.append(y_pred.cpu().numpy())

    return np.concatenate(predictions)


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
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> tuple[nn.Module, list[dict], float]:
    criterion = nn.CrossEntropyLoss()

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

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        running_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(y_batch.shape[0])
            running_samples += int(y_batch.shape[0])

        train_epoch_nll = running_loss / running_samples

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            device=device,
            num_classes=num_classes,
        )

        validation_nll = validation_metrics["negative_log_likelihood"]

        epoch_result = {
            "epoch": epoch,
            "train_epoch_nll": float(train_epoch_nll),
            "validation_nll": float(validation_nll),
            "validation_accuracy": float(validation_metrics["accuracy"]),
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:>3} | "
            f"train NLL={train_epoch_nll:.4f} | "
            f"val NLL={validation_nll:.4f} | "
            f"val acc={100 * validation_metrics['accuracy']:.2f}% | "
            f"val macro-F1={validation_metrics['macro_f1']:.4f}"
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


def main() -> None:
    set_seed(42)

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
    input_dim = int(X.shape[1])
    num_classes = int(metadata["num_classes"])
    clip_ticks = int(metadata["clip_ticks"])

    train_idx, validation_idx, test_idx = chronological_split(num_samples)

    X_train = X[train_idx]
    X_validation = X[validation_idx]
    X_test = X[test_idx]

    y_train = y_class[train_idx]
    y_validation = y_class[validation_idx]
    y_test = y_class[test_idx]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_validation_scaled = scaler.transform(X_validation).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    batch_size = 512
    hidden_dims = [128, 128]
    dropout = 0.2
    learning_rate = 1e-3
    weight_decay = 1e-4
    max_epochs = 50
    patience = 6

    train_loader = make_loader(
        X=X_train_scaled,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = make_loader(
        X=X_validation_scaled,
        y=y_validation,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = make_loader(
        X=X_test_scaled,
        y=y_test,
        batch_size=batch_size,
        shuffle=False,
    )

    train_eval_loader = make_loader(
        X=X_train_scaled,
        y=y_train,
        batch_size=batch_size,
        shuffle=False,
    )

    device = get_device()

    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    print("Standard neural network baseline")
    print("================================")
    print(f"Device:                  {device}")
    print(f"Number of samples:       {num_samples:,}")
    print(f"Training samples:        {len(train_idx):,}")
    print(f"Validation samples:      {len(validation_idx):,}")
    print(f"Test samples:            {len(test_idx):,}")
    print(f"Number of features:      {input_dim}")
    print(f"Number of classes:       {num_classes}")
    print(f"Hidden layers:           {hidden_dims}")
    print(f"Dropout:                 {dropout}")
    print(f"Learning rate:           {learning_rate}")
    print(f"Weight decay:            {weight_decay}")
    print()

    model, history, training_time = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
    )

    print()
    print(f"Training time:           {training_time:.2f} seconds")
    print()

    train_metrics = evaluate_model(
        model=model,
        data_loader=train_eval_loader,
        device=device,
        num_classes=num_classes,
    )

    validation_metrics = evaluate_model(
        model=model,
        data_loader=validation_loader,
        device=device,
        num_classes=num_classes,
    )

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=num_classes,
    )

    test_predictions = get_predictions(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print_most_common_predictions(
        y_pred=test_predictions,
        clip_ticks=clip_ticks,
    )

    print()
    print("Final metrics for best validation model")
    print("---------------------------------------")
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
        "model": "standard_mlp",
        "metadata": metadata,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "training_time_seconds": float(training_time),
        "history": history,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    output_path = results_dir / "standard_mlp.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    model_path = results_dir / "standard_mlp.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "metadata": metadata,
        },
        model_path,
    )

    print()
    print(f"Saved results to: {output_path}")
    print(f"Saved model to:   {model_path}")


if __name__ == "__main__":
    main()