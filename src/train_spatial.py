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
    y_pair = np.load(processed_dir / "y_pair.npy", mmap_mode="r")

    with open(processed_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    return X, y_pair, metadata


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


class SpatialJointNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        clip_ticks: int,
        hidden_dims: list[int],
        bid_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if clip_ticks < 1:
            raise ValueError("clip_ticks must be at least 1.")

        self.clip_ticks = int(clip_ticks)
        self.num_values = 2 * self.clip_ticks + 1
        self.num_hazards = max(self.clip_ticks - 1, 0)

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)
        encoded_dim = previous_dim

        self.ask_sign = nn.Linear(encoded_dim, 3)

        if self.num_hazards > 0:
            self.ask_pos_hazard = nn.Linear(encoded_dim, self.num_hazards)
            self.ask_neg_hazard = nn.Linear(encoded_dim, self.num_hazards)
        else:
            self.ask_pos_hazard = None
            self.ask_neg_hazard = None

        bid_input_dim = encoded_dim + self.num_values + 1

        self.bid_encoder = nn.Sequential(
            nn.Linear(bid_input_dim, bid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bid_hidden_dim, bid_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.bid_sign = nn.Linear(bid_hidden_dim, 3)

        if self.num_hazards > 0:
            self.bid_pos_hazard = nn.Linear(bid_hidden_dim, self.num_hazards)
            self.bid_neg_hazard = nn.Linear(bid_hidden_dim, self.num_hazards)
        else:
            self.bid_pos_hazard = None
            self.bid_neg_hazard = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def ask_distribution_parameters(
        self,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        sign_logits = self.ask_sign(encoded)

        if self.num_hazards > 0:
            pos_logits = self.ask_pos_hazard(encoded)
            neg_logits = self.ask_neg_hazard(encoded)
        else:
            pos_logits = None
            neg_logits = None

        return sign_logits, pos_logits, neg_logits

    def bid_distribution_parameters(
        self,
        encoded: torch.Tensor,
        ask_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        ask_class = ask_value + self.clip_ticks

        ask_one_hot = F.one_hot(
            ask_class,
            num_classes=self.num_values,
        ).float()

        ask_scaled = ask_value.float().unsqueeze(1) / float(self.clip_ticks)

        bid_input = torch.cat(
            [encoded, ask_one_hot, ask_scaled],
            dim=1,
        )

        bid_hidden = self.bid_encoder(bid_input)
        sign_logits = self.bid_sign(bid_hidden)

        if self.num_hazards > 0:
            pos_logits = self.bid_pos_hazard(bid_hidden)
            neg_logits = self.bid_neg_hazard(bid_hidden)
        else:
            pos_logits = None
            neg_logits = None

        return sign_logits, pos_logits, neg_logits

    def log_probs_one_dim_all_values(
        self,
        sign_logits: torch.Tensor,
        pos_logits: torch.Tensor | None,
        neg_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = sign_logits.shape[0]
        device = sign_logits.device
        K = self.clip_ticks
        V = self.num_values

        log_sign_probs = F.log_softmax(sign_logits, dim=1)

        out = torch.empty(batch_size, V, device=device)

        out[:, K] = log_sign_probs[:, 1]

        if K == 1:
            out[:, K + 1] = log_sign_probs[:, 2]
            out[:, K - 1] = log_sign_probs[:, 0]
            return out

        assert pos_logits is not None
        assert neg_logits is not None

        pos_survival = F.logsigmoid(-pos_logits)
        pos_stop = F.logsigmoid(pos_logits)

        neg_survival = F.logsigmoid(-neg_logits)
        neg_stop = F.logsigmoid(neg_logits)

        zeros = torch.zeros(batch_size, 1, device=device)

        pos_exclusive_survival = torch.cat(
            [zeros, torch.cumsum(pos_survival[:, :-1], dim=1)],
            dim=1,
        )

        neg_exclusive_survival = torch.cat(
            [zeros, torch.cumsum(neg_survival[:, :-1], dim=1)],
            dim=1,
        )

        pos_nonboundary = (
            log_sign_probs[:, 2].unsqueeze(1)
            + pos_exclusive_survival
            + pos_stop
        )

        neg_nonboundary = (
            log_sign_probs[:, 0].unsqueeze(1)
            + neg_exclusive_survival
            + neg_stop
        )

        out[:, K + 1: K + K] = pos_nonboundary
        out[:, 1:K] = torch.flip(neg_nonboundary, dims=[1])

        out[:, 2 * K] = (
            log_sign_probs[:, 2]
            + pos_survival.sum(dim=1)
        )

        out[:, 0] = (
            log_sign_probs[:, 0]
            + neg_survival.sum(dim=1)
        )

        return out

    def log_prob_one_dim(
        self,
        y: torch.Tensor,
        sign_logits: torch.Tensor,
        pos_logits: torch.Tensor | None,
        neg_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        all_log_probs = self.log_probs_one_dim_all_values(
            sign_logits=sign_logits,
            pos_logits=pos_logits,
            neg_logits=neg_logits,
        )

        class_index = y + self.clip_ticks

        return all_log_probs.gather(
            dim=1,
            index=class_index.unsqueeze(1),
        ).squeeze(1)

    def log_prob_pair(
        self,
        x: torch.Tensor,
        y_pair: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(x)

        ask = y_pair[:, 0]
        bid = y_pair[:, 1]

        ask_sign, ask_pos, ask_neg = self.ask_distribution_parameters(encoded)

        log_prob_ask = self.log_prob_one_dim(
            y=ask,
            sign_logits=ask_sign,
            pos_logits=ask_pos,
            neg_logits=ask_neg,
        )

        bid_sign, bid_pos, bid_neg = self.bid_distribution_parameters(
            encoded=encoded,
            ask_value=ask,
        )

        log_prob_bid = self.log_prob_one_dim(
            y=bid,
            sign_logits=bid_sign,
            pos_logits=bid_pos,
            neg_logits=bid_neg,
        )

        return log_prob_ask + log_prob_bid

    def joint_log_prob_grid(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode(x)

        batch_size = x.shape[0]
        device = x.device
        K = self.clip_ticks
        V = self.num_values

        values = torch.arange(
            -K,
            K + 1,
            device=device,
            dtype=torch.long,
        )

        ask_sign, ask_pos, ask_neg = self.ask_distribution_parameters(encoded)

        ask_log_probs_all = self.log_probs_one_dim_all_values(
            sign_logits=ask_sign,
            pos_logits=ask_pos,
            neg_logits=ask_neg,
        )

        encoded_repeated = (
            encoded.unsqueeze(1)
            .expand(batch_size, V, encoded.shape[1])
            .reshape(batch_size * V, encoded.shape[1])
        )

        ask_values_repeated = (
            values.unsqueeze(0)
            .expand(batch_size, V)
            .reshape(batch_size * V)
        )

        bid_sign, bid_pos, bid_neg = self.bid_distribution_parameters(
            encoded=encoded_repeated,
            ask_value=ask_values_repeated,
        )

        bid_log_probs_all = self.log_probs_one_dim_all_values(
            sign_logits=bid_sign,
            pos_logits=bid_pos,
            neg_logits=bid_neg,
        )

        bid_log_probs_all = bid_log_probs_all.reshape(batch_size, V, V)

        joint_log_probs = (
            ask_log_probs_all.unsqueeze(2)
            + bid_log_probs_all
        )

        return joint_log_probs.reshape(batch_size, V * V)


def evaluate_nll_metrics(
    model: SpatialJointNN,
    X: np.ndarray,
    y_pair: np.ndarray,
    scaler: StandardScaler,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    movement_loss = 0.0
    movement_samples = 0

    with torch.no_grad():
        for batch_slice in iter_sequential_batches(start, end, batch_size):
            X_batch_np = transform_batch(X[batch_slice], scaler)
            y_batch_np = np.asarray(y_pair[batch_slice], dtype=np.int64)

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            log_prob = model.log_prob_pair(X_batch, y_batch)
            losses = -log_prob

            total_loss += float(losses.sum().item())
            total_samples += int(y_batch.shape[0])

            movement_mask = (y_batch[:, 0] != 0) | (y_batch[:, 1] != 0)

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
    model: SpatialJointNN,
    X: np.ndarray,
    y_pair: np.ndarray,
    scaler: StandardScaler,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
    num_classes: int,
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
            y_batch_np = np.asarray(y_pair[batch_slice], dtype=np.int64)

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            log_prob = model.log_prob_pair(X_batch, y_batch)
            losses = -log_prob

            total_loss += float(losses.sum().item())
            total_samples += int(y_batch.shape[0])

            movement_mask = (y_batch[:, 0] != 0) | (y_batch[:, 1] != 0)

            if movement_mask.any():
                movement_loss += float(losses[movement_mask].sum().item())
                movement_samples += int(movement_mask.sum().item())

            joint_log_probs = model.joint_log_prob_grid(X_batch)
            y_pred = torch.argmax(joint_log_probs, dim=1)

            # accumulate only movement observations.
            if movement_mask.any():
                joint_probs = torch.exp(joint_log_probs[movement_mask])
                movement_prob_sum += joint_probs.sum(dim=0).cpu().numpy()
                movement_prob_samples += int(movement_mask.sum().item())

            y_true = (
                (y_batch[:, 0] + model.clip_ticks) * model.num_values
                + (y_batch[:, 1] + model.clip_ticks)
            )

            all_y_true.append(y_true.cpu().numpy())
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
    model: SpatialJointNN,
    X: np.ndarray,
    y_pair: np.ndarray,
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
) -> tuple[SpatialJointNN, list[dict], float]:
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
            y_batch_np = np.asarray(y_pair[batch_indices], dtype=np.int64)

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            log_prob = model.log_prob_pair(X_batch, y_batch)
            loss = -log_prob.mean()

            loss.backward()
            optimizer.step()

            current_batch_size = int(y_batch.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            running_samples += current_batch_size

        train_epoch_nll = running_loss / running_samples

        validation_metrics = evaluate_nll_metrics(
            model=model,
            X=X,
            y_pair=y_pair,
            scaler=scaler,
            start=validation_start,
            end=validation_end,
            batch_size=eval_batch_size,
            device=device,
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
        description="Train reduced spatial neural network."
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/WSELOB_KGHM_h30_L50_full_tick5",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/WSELOB_KGHM_h30_L50_full_tick5_5days",
    )

    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--bid-hidden-dim", type=int, default=128)
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

    X, y_pair, metadata = load_dataset(processed_dir)

    num_samples = len(y_pair)
    input_dim = int(X.shape[1])
    clip_ticks = int(metadata["clip_ticks"])
    num_classes = int(metadata["num_classes"])

    n_train, n_validation, n_test = chronological_split(num_samples)

    train_start = 0
    train_end = n_train

    validation_start = n_train
    validation_end = n_train + n_validation

    test_start = validation_end
    test_end = num_samples

    device = get_device()

    print("Reduced spatial neural network")
    print("==============================")
    print(f"Device:                  {device}")
    print(f"Processed directory:     {processed_dir}")
    print(f"Results directory:       {results_dir}")
    print(f"Number of samples:       {num_samples:,}")
    print(f"Training samples:        {n_train:,}")
    print(f"Validation samples:      {n_validation:,}")
    print(f"Test samples:            {n_test:,}")
    print(f"Number of features:      {input_dim}")
    print(f"Clip ticks:              {clip_ticks}")
    print(f"Number of classes:       {num_classes}")
    print(f"Hidden layers:           {args.hidden_dims}")
    print(f"Bid hidden dimension:    {args.bid_hidden_dim}")
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

    model = SpatialJointNN(
        input_dim=input_dim,
        clip_ticks=clip_ticks,
        hidden_dims=args.hidden_dims,
        bid_hidden_dim=args.bid_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    model, history, training_time = train_model(
        model=model,
        X=X,
        y_pair=y_pair,
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
    )

    print()
    print(f"Training time:           {training_time:.2f} seconds")
    print()

    train_metrics = evaluate_nll_metrics(
        model=model,
        X=X,
        y_pair=y_pair,
        scaler=scaler,
        start=train_start,
        end=train_end,
        batch_size=args.eval_batch_size,
        device=device,
    )

    validation_metrics = evaluate_nll_metrics(
        model=model,
        X=X,
        y_pair=y_pair,
        scaler=scaler,
        start=validation_start,
        end=validation_end,
        batch_size=args.eval_batch_size,
        device=device,
    )

    test_metrics, test_predictions, test_avg_probs_movement = evaluate_full(
        model=model,
        X=X,
        y_pair=y_pair,
        scaler=scaler,
        start=test_start,
        end=test_end,
        batch_size=args.eval_batch_size,
        device=device,
        num_classes=num_classes,
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
        "model": "reduced_spatial_neural_network",
        "metadata": metadata,
        "input_dim": input_dim,
        "clip_ticks": clip_ticks,
        "num_classes": num_classes,
        "hidden_dims": args.hidden_dims,
        "bid_hidden_dim": args.bid_hidden_dim,
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

    output_path = results_dir / "spatial_nn.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    avg_probs_path = results_dir / "spatial_nn_avg_probs_movement_test.npy"

    np.save(
        avg_probs_path,
        test_avg_probs_movement,
    )

    model_path = results_dir / "spatial_nn.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "clip_ticks": clip_ticks,
            "hidden_dims": args.hidden_dims,
            "bid_hidden_dim": args.bid_hidden_dim,
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