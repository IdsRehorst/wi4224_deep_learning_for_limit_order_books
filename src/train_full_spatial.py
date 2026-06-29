from pathlib import Path
import argparse
import json
import time
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root() / path


# ---------------------------------------------------------------------
# Reproducibility / device
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_spatial_dataset(processed_dir: Path) -> tuple[dict, dict]:
    spatial_dir = processed_dir / "spatial"

    if not spatial_dir.exists():
        raise FileNotFoundError(
            f"Could not find spatial directory: {spatial_dir}\n"
            f"Run convert_wselob.py with --save-dense-grid first."
        )

    required_files = {
        "ask_dense_log_size": spatial_dir / "ask_dense_log_size.npy",
        "bid_dense_log_size": spatial_dir / "bid_dense_log_size.npy",
        "spread_ticks": spatial_dir / "spread_ticks.npy",
        "y_pair": spatial_dir / "y_pair.npy",
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required spatial file: {path}\n"
                f"Tip: run convert_wselob.py with --save-dense-grid."
            )

    arrays = {
        "ask_dense_log_size": np.load(
            required_files["ask_dense_log_size"],
            mmap_mode="r",
        ),
        "bid_dense_log_size": np.load(
            required_files["bid_dense_log_size"],
            mmap_mode="r",
        ),
        "spread_ticks": np.load(
            required_files["spread_ticks"],
            mmap_mode="r",
        ),
        "y_pair": np.load(
            required_files["y_pair"],
            mmap_mode="r",
        ),
    }

    metadata_path = processed_dir / "metadata.json"
    spatial_metadata_path = spatial_dir / "spatial_metadata.json"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    with open(spatial_metadata_path, "r") as f:
        spatial_metadata = json.load(f)

    metadata["spatial_metadata"] = spatial_metadata

    return arrays, metadata


def chronological_split(
    n_samples: int,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[int, int, int]:
    n_train = int(train_fraction * n_samples)
    n_validation = int(validation_fraction * n_samples)
    n_test = n_samples - n_train - n_validation

    return n_train, n_validation, n_test


# ---------------------------------------------------------------------
# Batch iteration
# ---------------------------------------------------------------------

def iter_sequential_batches(start: int, end: int, batch_size: int):
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        yield slice(batch_start, batch_end)


def iter_chunk_shuffled_slices(
    start: int,
    end: int,
    batch_size: int,
    chunk_size: int,
    rng: np.random.Generator,
):
    """
    Shuffle large contiguous chunks, then read sequentially inside chunks.

    This avoids slow fully-random indexing into memory-mapped arrays.
    """
    if chunk_size < batch_size:
        raise ValueError("chunk_size must be at least as large as batch_size.")

    chunk_starts = np.arange(start, end, chunk_size)
    rng.shuffle(chunk_starts)

    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + chunk_size, end)

        for batch_start in range(chunk_start, chunk_end, batch_size):
            batch_end = min(batch_start + batch_size, chunk_end)
            yield slice(batch_start, batch_end)


# ---------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------

def fit_spatial_scalers(
    arrays: dict,
    train_end: int,
    batch_size: int,
) -> dict:
    ask_scaler = StandardScaler()
    bid_scaler = StandardScaler()
    spread_scaler = StandardScaler()

    ask_dense = arrays["ask_dense_log_size"]
    bid_dense = arrays["bid_dense_log_size"]
    spread = arrays["spread_ticks"]

    for batch_slice in iter_sequential_batches(0, train_end, batch_size):
        ask_batch = np.asarray(ask_dense[batch_slice], dtype=np.float32)
        bid_batch = np.asarray(bid_dense[batch_slice], dtype=np.float32)
        spread_batch = np.asarray(
            spread[batch_slice],
            dtype=np.float32,
        ).reshape(-1, 1)

        ask_scaler.partial_fit(ask_batch)
        bid_scaler.partial_fit(bid_batch)
        spread_scaler.partial_fit(spread_batch)

    return {
        "ask": ask_scaler,
        "bid": bid_scaler,
        "spread": spread_scaler,
    }


def transform_spatial_batch(
    arrays: dict,
    batch_slice,
    scalers: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ask_dense = np.asarray(
        arrays["ask_dense_log_size"][batch_slice],
        dtype=np.float32,
    )

    bid_dense = np.asarray(
        arrays["bid_dense_log_size"][batch_slice],
        dtype=np.float32,
    )

    spread = np.asarray(
        arrays["spread_ticks"][batch_slice],
        dtype=np.float32,
    ).reshape(-1, 1)

    # Important: copy avoids PyTorch warning about non-writable memmap views.
    y_pair = np.asarray(
        arrays["y_pair"][batch_slice],
        dtype=np.int64,
    ).copy()

    ask_scaled = scalers["ask"].transform(ask_dense).astype(np.float32)
    bid_scaled = scalers["bid"].transform(bid_dense).astype(np.float32)
    spread_scaled = scalers["spread"].transform(spread).astype(np.float32)

    return (
        np.ascontiguousarray(ask_scaled),
        np.ascontiguousarray(bid_scaled),
        np.ascontiguousarray(spread_scaled.reshape(-1)),
        np.ascontiguousarray(y_pair),
    )


# ---------------------------------------------------------------------
# Class helpers
# ---------------------------------------------------------------------

def decode_class(class_id: int, clip_ticks: int) -> tuple[int, int]:
    num_values = 2 * clip_ticks + 1

    delta_ask = class_id // num_values - clip_ticks
    delta_bid = class_id % num_values - clip_ticks

    return int(delta_ask), int(delta_bid)


# ---------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------

def make_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers = []
    previous_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        previous_dim = hidden_dim

    layers.append(nn.Linear(previous_dim, output_dim))

    return nn.Sequential(*layers)


def clone_state_dict(model: nn.Module) -> dict:
    """
    Clone model state dict to CPU.

    Used for storing best checkpoints during training.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


# ---------------------------------------------------------------------
# Full local spatial neural network
# ---------------------------------------------------------------------

class FullSpatialJointNN(nn.Module):
    """
    Local spatial neural network for joint best ask/bid movement prediction.

    Factorization:

        p(Delta A, Delta B | X)
        =
        p(Delta A | X) p(Delta B | Delta A, X)

    For each dimension, the model predicts:
        1. sign: negative / zero / positive
        2. magnitude via local continuation probabilities.

    Local continuation probabilities are computed from windows around
    candidate tick offsets in the dense ask/bid order book.
    """

    def __init__(
        self,
        dense_depth: int,
        clip_ticks: int,
        global_depth: int = 50,
        local_window: int = 2,
        encoder_hidden_dims: list[int] | None = None,
        local_hidden_dims: list[int] | None = None,
        bid_hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if clip_ticks < 2:
            raise ValueError("clip_ticks must be at least 2 for this model.")

        if encoder_hidden_dims is None:
            encoder_hidden_dims = [128, 128]

        if local_hidden_dims is None:
            local_hidden_dims = [128, 64]

        if bid_hidden_dims is None:
            bid_hidden_dims = [128, 128]

        self.dense_depth = int(dense_depth)
        self.clip_ticks = int(clip_ticks)
        self.num_values = 2 * self.clip_ticks + 1
        self.num_hazards = self.clip_ticks - 1
        self.global_depth = min(int(global_depth), int(dense_depth))
        self.local_window = int(local_window)
        self.local_width = 2 * self.local_window + 1

        global_input_dim = 2 * self.global_depth + 1

        self.global_encoder = make_mlp(
            input_dim=global_input_dim,
            hidden_dims=encoder_hidden_dims[:-1],
            output_dim=encoder_hidden_dims[-1],
            dropout=dropout,
        )

        encoded_dim = encoder_hidden_dims[-1]

        self.ask_sign = nn.Linear(encoded_dim, 3)

        bid_context_input_dim = encoded_dim + self.num_values + 1

        self.bid_context_encoder = make_mlp(
            input_dim=bid_context_input_dim,
            hidden_dims=bid_hidden_dims[:-1],
            output_dim=bid_hidden_dims[-1],
            dropout=dropout,
        )

        bid_context_dim = bid_hidden_dims[-1]

        self.bid_sign = nn.Linear(bid_context_dim, 3)

        ask_local_input_dim = (
            2 * self.local_width
            + 1
            + 1
            + encoded_dim
        )

        bid_local_input_dim = (
            2 * self.local_width
            + 1
            + 1
            + bid_context_dim
        )

        self.ask_positive_continue = make_mlp(
            input_dim=ask_local_input_dim,
            hidden_dims=local_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

        self.ask_negative_continue = make_mlp(
            input_dim=ask_local_input_dim,
            hidden_dims=local_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

        self.bid_positive_continue = make_mlp(
            input_dim=bid_local_input_dim,
            hidden_dims=local_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

        self.bid_negative_continue = make_mlp(
            input_dim=bid_local_input_dim,
            hidden_dims=local_hidden_dims,
            output_dim=1,
            dropout=dropout,
        )

        # Candidate offsets are 1,...,K-1 for non-boundary magnitudes.
        candidate_offsets = torch.arange(1, self.clip_ticks, dtype=torch.long)
        relative_window = torch.arange(
            -self.local_window,
            self.local_window + 1,
            dtype=torch.long,
        )

        # Padding shifts every original index by local_window.
        local_indices = (
            candidate_offsets[:, None]
            + relative_window[None, :]
            + self.local_window
        )

        padded_width = self.dense_depth + 2 * self.local_window
        local_indices = torch.clamp(local_indices, 0, padded_width - 1)

        self.register_buffer(
            "local_indices_flat",
            local_indices.reshape(-1),
            persistent=False,
        )

    def encode_global(
        self,
        ask_dense: torch.Tensor,
        bid_dense: torch.Tensor,
        spread: torch.Tensor,
    ) -> torch.Tensor:
        global_features = torch.cat(
            [
                ask_dense[:, : self.global_depth],
                bid_dense[:, : self.global_depth],
                spread.unsqueeze(1),
            ],
            dim=1,
        )

        return self.global_encoder(global_features)

    def make_bid_context(
        self,
        encoded: torch.Tensor,
        ask_value: torch.Tensor,
    ) -> torch.Tensor:
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

        return self.bid_context_encoder(bid_input)

    def local_windows(self, dense: torch.Tensor) -> torch.Tensor:
        """
        Extract local windows around candidate offsets 1,...,K-1.

        Input:
            dense: (B, dense_depth)

        Output:
            windows: (B, K-1, 2w+1)
        """
        padded = F.pad(
            dense,
            pad=(self.local_window, self.local_window),
            mode="constant",
            value=0.0,
        )

        selected = torch.index_select(padded,dim=1,index=self.local_indices_flat,)

        batch_size = dense.shape[0]

        return selected.reshape(
            batch_size,
            self.num_hazards,
            self.local_width,
        )

    def continuation_logits(
        self,
        context: torch.Tensor,
        ask_dense: torch.Tensor,
        bid_dense: torch.Tensor,
        spread: torch.Tensor,
        continuation_net: nn.Module,
    ) -> torch.Tensor:
        """
        Compute continuation logits for candidate levels s=1,...,K-1.

        Output:
            logits: (B, K-1)
        """
        batch_size = ask_dense.shape[0]
        device = ask_dense.device

        ask_windows = self.local_windows(ask_dense)
        bid_windows = self.local_windows(bid_dense)

        spread_features = spread.view(batch_size, 1, 1).expand(
            batch_size,
            self.num_hazards,
            1,
        )

        offset_features = (
            torch.arange(
                1,
                self.clip_ticks,
                device=device,
                dtype=torch.float32,
            )
            .view(1, self.num_hazards, 1)
            .expand(batch_size, self.num_hazards, 1)
            / float(self.clip_ticks)
        )

        context_features = context.unsqueeze(1).expand(
            batch_size,
            self.num_hazards,
            context.shape[1],
        )

        local_features = torch.cat(
            [
                ask_windows,
                bid_windows,
                spread_features,
                offset_features,
                context_features,
            ],
            dim=2,
        )

        flat_features = local_features.reshape(
            batch_size * self.num_hazards,
            -1,
        )

        logits = continuation_net(flat_features).reshape(
            batch_size,
            self.num_hazards,
        )

        return logits

    def log_probs_from_components(
        self,
        sign_logits: torch.Tensor,
        positive_continue_logits: torch.Tensor,
        negative_continue_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert sign logits and continuation logits to log probabilities over
        values {-K,...,K}.

        Continuation convention:
            c_s = P(|Y| > s | |Y| >= s, sign, X)

        For r=1,...,K-1:
            P(|Y|=r) = prod_{s<r} c_s * (1-c_r)

        Boundary:
            P(|Y|=K) = prod_{s=1}^{K-1} c_s
        """
        batch_size = sign_logits.shape[0]
        device = sign_logits.device
        K = self.clip_ticks
        V = self.num_values

        log_sign_probs = F.log_softmax(sign_logits, dim=1)

        positive_continue_log = F.logsigmoid(positive_continue_logits)
        positive_stop_log = F.logsigmoid(-positive_continue_logits)

        negative_continue_log = F.logsigmoid(negative_continue_logits)
        negative_stop_log = F.logsigmoid(-negative_continue_logits)

        out = torch.empty(batch_size, V, device=device)

        # Zero movement.
        out[:, K] = log_sign_probs[:, 1]

        zeros = torch.zeros(batch_size, 1, device=device)

        positive_exclusive_continue = torch.cat(
            [
                zeros,
                torch.cumsum(positive_continue_log[:, :-1], dim=1),
            ],
            dim=1,
        )

        negative_exclusive_continue = torch.cat(
            [
                zeros,
                torch.cumsum(negative_continue_log[:, :-1], dim=1),
            ],
            dim=1,
        )

        positive_nonboundary = (
            log_sign_probs[:, 2].unsqueeze(1)
            + positive_exclusive_continue
            + positive_stop_log
        )

        negative_nonboundary = (
            log_sign_probs[:, 0].unsqueeze(1)
            + negative_exclusive_continue
            + negative_stop_log
        )

        # Positive r=1,...,K-1 maps to indices K+1,...,2K-1.
        out[:, K + 1: K + K] = positive_nonboundary

        # Negative r=1,...,K-1 maps to indices K-1,...,1.
        out[:, 1:K] = torch.flip(negative_nonboundary, dims=[1])

        # Boundary classes +K and -K.
        out[:, 2 * K] = (
            log_sign_probs[:, 2]
            + positive_continue_log.sum(dim=1)
        )

        out[:, 0] = (
            log_sign_probs[:, 0]
            + negative_continue_log.sum(dim=1)
        )

        return out

    def ask_log_probs_all_values(
        self,
        encoded: torch.Tensor,
        ask_dense: torch.Tensor,
        bid_dense: torch.Tensor,
        spread: torch.Tensor,
    ) -> torch.Tensor:
        sign_logits = self.ask_sign(encoded)

        positive_continue_logits = self.continuation_logits(
            context=encoded,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
            continuation_net=self.ask_positive_continue,
        )

        negative_continue_logits = self.continuation_logits(
            context=encoded,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
            continuation_net=self.ask_negative_continue,
        )

        return self.log_probs_from_components(
            sign_logits=sign_logits,
            positive_continue_logits=positive_continue_logits,
            negative_continue_logits=negative_continue_logits,
        )

    def bid_log_probs_all_values(
        self,
        bid_context: torch.Tensor,
        ask_dense: torch.Tensor,
        bid_dense: torch.Tensor,
        spread: torch.Tensor,
    ) -> torch.Tensor:
        sign_logits = self.bid_sign(bid_context)

        positive_continue_logits = self.continuation_logits(
            context=bid_context,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
            continuation_net=self.bid_positive_continue,
        )

        negative_continue_logits = self.continuation_logits(
            context=bid_context,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
            continuation_net=self.bid_negative_continue,
        )

        return self.log_probs_from_components(
            sign_logits=sign_logits,
            positive_continue_logits=positive_continue_logits,
            negative_continue_logits=negative_continue_logits,
        )

    def log_prob_pair(
        self,
        ask_dense: torch.Tensor,
        bid_dense: torch.Tensor,
        spread: torch.Tensor,
        y_pair: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode_global(
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
        )

        ask_values = y_pair[:, 0]
        bid_values = y_pair[:, 1]

        ask_log_probs = self.ask_log_probs_all_values(
            encoded=encoded,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
        )

        ask_indices = ask_values + self.clip_ticks

        log_prob_ask = ask_log_probs.gather(
            dim=1,
            index=ask_indices.unsqueeze(1),
        ).squeeze(1)

        bid_context = self.make_bid_context(
            encoded=encoded,
            ask_value=ask_values,
        )

        bid_log_probs = self.bid_log_probs_all_values(
            bid_context=bid_context,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
        )

        bid_indices = bid_values + self.clip_ticks

        log_prob_bid = bid_log_probs.gather(
            dim=1,
            index=bid_indices.unsqueeze(1),
        ).squeeze(1)

        return log_prob_ask + log_prob_bid

    def joint_log_prob_grid(
        self,
        ask_dense: torch.Tensor,
        bid_dense: torch.Tensor,
        spread: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full joint log probability grid.
        """
        encoded = self.encode_global(
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
        )

        batch_size = ask_dense.shape[0]
        device = ask_dense.device
        K = self.clip_ticks
        V = self.num_values

        ask_log_probs = self.ask_log_probs_all_values(
            encoded=encoded,
            ask_dense=ask_dense,
            bid_dense=bid_dense,
            spread=spread,
        )

        ask_values = torch.arange(
            -K,
            K + 1,
            device=device,
            dtype=torch.long,
        )

        encoded_repeated = (
            encoded.unsqueeze(1)
            .expand(batch_size, V, encoded.shape[1])
            .reshape(batch_size * V, encoded.shape[1])
        )

        ask_values_repeated = (
            ask_values.unsqueeze(0)
            .expand(batch_size, V)
            .reshape(batch_size * V)
        )

        ask_dense_repeated = (
            ask_dense.unsqueeze(1)
            .expand(batch_size, V, ask_dense.shape[1])
            .reshape(batch_size * V, ask_dense.shape[1])
        )

        bid_dense_repeated = (
            bid_dense.unsqueeze(1)
            .expand(batch_size, V, bid_dense.shape[1])
            .reshape(batch_size * V, bid_dense.shape[1])
        )

        spread_repeated = (
            spread.unsqueeze(1)
            .expand(batch_size, V)
            .reshape(batch_size * V)
        )

        bid_context = self.make_bid_context(
            encoded=encoded_repeated,
            ask_value=ask_values_repeated,
        )

        bid_log_probs = self.bid_log_probs_all_values(
            bid_context=bid_context,
            ask_dense=ask_dense_repeated,
            bid_dense=bid_dense_repeated,
            spread=spread_repeated,
        )

        bid_log_probs = bid_log_probs.reshape(batch_size, V, V)

        joint_log_probs = (
            ask_log_probs.unsqueeze(2)
            + bid_log_probs
        )

        return joint_log_probs.reshape(batch_size, V * V)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_nll_metrics(
    model: FullSpatialJointNN,
    arrays: dict,
    scalers: dict,
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
            ask_np, bid_np, spread_np, y_np = transform_spatial_batch(
                arrays=arrays,
                batch_slice=batch_slice,
                scalers=scalers,
            )

            ask = torch.from_numpy(ask_np).to(device)
            bid = torch.from_numpy(bid_np).to(device)
            spread = torch.from_numpy(spread_np).to(device)
            y_pair = torch.as_tensor(y_np, dtype=torch.long, device=device)

            log_prob = model.log_prob_pair(
                ask_dense=ask,
                bid_dense=bid,
                spread=spread,
                y_pair=y_pair,
            )

            losses = -log_prob

            total_loss += float(losses.sum().item())
            total_samples += int(y_pair.shape[0])

            movement_mask = (y_pair[:, 0] != 0) | (y_pair[:, 1] != 0)

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
    model: FullSpatialJointNN,
    arrays: dict,
    scalers: dict,
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

    # accumulate the average predicted distribution on movement
    # observations only. This gives a compact length-441 vector that can be
    # reshaped into a heatmap without saving all per-sample predictions.
    movement_prob_sum = np.zeros(num_classes, dtype=np.float64)
    movement_prob_samples = 0

    with torch.no_grad():
        for batch_slice in iter_sequential_batches(start, end, batch_size):
            ask_np, bid_np, spread_np, y_np = transform_spatial_batch(
                arrays=arrays,
                batch_slice=batch_slice,
                scalers=scalers,
            )

            ask = torch.from_numpy(ask_np).to(device)
            bid = torch.from_numpy(bid_np).to(device)
            spread = torch.from_numpy(spread_np).to(device)
            y_pair = torch.as_tensor(y_np, dtype=torch.long, device=device)

            log_prob = model.log_prob_pair(
                ask_dense=ask,
                bid_dense=bid,
                spread=spread,
                y_pair=y_pair,
            )

            losses = -log_prob

            total_loss += float(losses.sum().item())
            total_samples += int(y_pair.shape[0])

            movement_mask = (y_pair[:, 0] != 0) | (y_pair[:, 1] != 0)

            if movement_mask.any():
                movement_loss += float(losses[movement_mask].sum().item())
                movement_count = int(movement_mask.sum().item())
                movement_samples += movement_count

            joint_log_probs = model.joint_log_prob_grid(
                ask_dense=ask,
                bid_dense=bid,
                spread=spread,
            )

            y_pred = torch.argmax(joint_log_probs, dim=1)

            # New: accumulate predicted probabilities on movement observations.
            # joint_log_probs is already a normalized log-probability grid over
            # the 441 joint movement classes, so exponentiating gives
            # probabilities.
            if movement_mask.any():
                movement_probs = torch.exp(joint_log_probs[movement_mask])
                movement_prob_sum += movement_probs.sum(dim=0).cpu().numpy()
                movement_prob_samples += movement_count

            y_true = (
                (y_pair[:, 0] + model.clip_ticks) * model.num_values
                + (y_pair[:, 1] + model.clip_ticks)
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


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_model(
    model: FullSpatialJointNN,
    arrays: dict,
    scalers: dict,
    train_end: int,
    validation_start: int,
    validation_end: int,
    batch_size: int,
    eval_batch_size: int,
    chunk_size: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict, list[dict], float, dict]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_nll = float("inf")
    best_validation_movement_nll = float("inf")

    best_validation_nll_state_dict = None
    best_validation_movement_nll_state_dict = None

    best_validation_nll_epoch = None
    best_validation_movement_nll_epoch = None

    epochs_without_overall_improvement = 0

    history = []
    start_time = time.time()

    rng = np.random.default_rng(seed)

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        running_samples = 0

        for batch_slice in iter_chunk_shuffled_slices(
            start=0,
            end=train_end,
            batch_size=batch_size,
            chunk_size=chunk_size,
            rng=rng,
        ):
            ask_np, bid_np, spread_np, y_np = transform_spatial_batch(
                arrays=arrays,
                batch_slice=batch_slice,
                scalers=scalers,
            )

            ask = torch.from_numpy(ask_np).to(device)
            bid = torch.from_numpy(bid_np).to(device)
            spread = torch.from_numpy(spread_np).to(device)
            y_pair = torch.as_tensor(y_np, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)

            log_prob = model.log_prob_pair(
                ask_dense=ask,
                bid_dense=bid,
                spread=spread,
                y_pair=y_pair,
            )

            loss = -log_prob.mean()

            loss.backward()
            optimizer.step()

            current_batch_size = int(y_pair.shape[0])
            running_loss += float(loss.item()) * current_batch_size
            running_samples += current_batch_size

        train_epoch_nll = running_loss / running_samples

        validation_metrics = evaluate_nll_metrics(
            model=model,
            arrays=arrays,
            scalers=scalers,
            start=validation_start,
            end=validation_end,
            batch_size=eval_batch_size,
            device=device,
        )

        validation_nll = validation_metrics["negative_log_likelihood"]
        validation_movement_nll = validation_metrics[
            "movement_negative_log_likelihood"
        ]

        epoch_result = {
            "epoch": epoch,
            "train_epoch_nll": float(train_epoch_nll),
            "validation_nll": float(validation_nll),
            "validation_movement_nll": float(validation_movement_nll),
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch:>3} | "
            f"train NLL={train_epoch_nll:.4f} | "
            f"val NLL={validation_nll:.4f} | "
            f"val move NLL={validation_movement_nll:.4f}"
        )

        improved_overall = validation_nll < best_validation_nll
        improved_movement = validation_movement_nll < best_validation_movement_nll

        if improved_overall:
            best_validation_nll = validation_nll
            best_validation_nll_epoch = epoch
            best_validation_nll_state_dict = clone_state_dict(model)
            epochs_without_overall_improvement = 0
        else:
            epochs_without_overall_improvement += 1

        if improved_movement:
            best_validation_movement_nll = validation_movement_nll
            best_validation_movement_nll_epoch = epoch
            best_validation_movement_nll_state_dict = clone_state_dict(model)

        if epochs_without_overall_improvement >= patience:
            print()
            print(f"Early stopping after {epoch} epochs.")
            break

    training_time = time.time() - start_time

    if best_validation_nll_state_dict is None:
        raise RuntimeError("No best-validation-NLL checkpoint was saved.")

    if best_validation_movement_nll_state_dict is None:
        raise RuntimeError("No best-validation-movement-NLL checkpoint was saved.")

    checkpoint_state_dicts = {
        "best_validation_nll": best_validation_nll_state_dict,
        "best_validation_movement_nll": best_validation_movement_nll_state_dict,
    }

    checkpoint_info = {
        "best_validation_nll_epoch": best_validation_nll_epoch,
        "best_validation_nll": float(best_validation_nll),
        "best_validation_movement_nll_epoch": best_validation_movement_nll_epoch,
        "best_validation_movement_nll": float(best_validation_movement_nll),
    }

    return checkpoint_state_dicts, history, training_time, checkpoint_info


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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train full local spatial neural network."
    )

    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed/WSELOB_KGHM_h30_L50_full_tick5",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/WSELOB_KGHM_h30_L50_full_tick5_full_spatial",
    )

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--full-eval-batch-size", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1_048_576)

    parser.add_argument("--global-depth", type=int, default=50)
    parser.add_argument("--local-window", type=int, default=2)

    parser.add_argument(
        "--encoder-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
    )

    parser.add_argument(
        "--local-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64],
    )

    parser.add_argument(
        "--bid-hidden-dims",
        type=int,
        nargs="+",
        default=[128, 128],
    )

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

    arrays, metadata = load_spatial_dataset(processed_dir)

    n_samples = int(arrays["y_pair"].shape[0])
    dense_depth = int(arrays["ask_dense_log_size"].shape[1])
    clip_ticks = int(metadata["clip_ticks"])
    num_classes = int(metadata["num_classes"])

    n_train, n_validation, n_test = chronological_split(n_samples)

    train_start = 0
    train_end = n_train

    validation_start = n_train
    validation_end = n_train + n_validation

    test_start = validation_end
    test_end = n_samples

    device = get_device()

    print("Full local spatial neural network")
    print("=================================")
    print(f"Device:                  {device}")
    print(f"Processed directory:     {processed_dir}")
    print(f"Results directory:       {results_dir}")
    print(f"Number of samples:       {n_samples:,}")
    print(f"Training samples:        {n_train:,}")
    print(f"Validation samples:      {n_validation:,}")
    print(f"Test samples:            {n_test:,}")
    print(f"Dense depth:             {dense_depth}")
    print(f"Clip ticks:              {clip_ticks}")
    print(f"Number of classes:       {num_classes}")
    print(f"Global depth:            {args.global_depth}")
    print(f"Local window:            {args.local_window}")
    print(f"Encoder hidden dims:     {args.encoder_hidden_dims}")
    print(f"Local hidden dims:       {args.local_hidden_dims}")
    print(f"Bid hidden dims:         {args.bid_hidden_dims}")
    print(f"Batch size:              {args.batch_size}")
    print(f"Eval batch size:         {args.eval_batch_size}")
    print(f"Full eval batch size:    {args.full_eval_batch_size}")
    print(f"Learning rate:           {args.learning_rate}")
    print(f"Weight decay:            {args.weight_decay}")
    print()

    print("Fitting spatial scalers...")
    scalers = fit_spatial_scalers(
        arrays=arrays,
        train_end=train_end,
        batch_size=args.eval_batch_size,
    )
    print("Scalers fitted.")
    print()

    model = FullSpatialJointNN(
        dense_depth=dense_depth,
        clip_ticks=clip_ticks,
        global_depth=args.global_depth,
        local_window=args.local_window,
        encoder_hidden_dims=args.encoder_hidden_dims,
        local_hidden_dims=args.local_hidden_dims,
        bid_hidden_dims=args.bid_hidden_dims,
        dropout=args.dropout,
    ).to(device)

    checkpoint_state_dicts, history, training_time, checkpoint_info = train_model(
        model=model,
        arrays=arrays,
        scalers=scalers,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        chunk_size=args.chunk_size,
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
    print("Checkpoint summary")
    print("------------------")
    print(
        f"Best validation NLL:       epoch "
        f"{checkpoint_info['best_validation_nll_epoch']} | "
        f"{checkpoint_info['best_validation_nll']:.4f}"
    )
    print(
        f"Best validation move NLL:  epoch "
        f"{checkpoint_info['best_validation_movement_nll_epoch']} | "
        f"{checkpoint_info['best_validation_movement_nll']:.4f}"
    )

    def evaluate_checkpoint(selection_name: str, state_dict: dict):
        model.load_state_dict(state_dict)

        train_metrics = evaluate_nll_metrics(
            model=model,
            arrays=arrays,
            scalers=scalers,
            start=train_start,
            end=train_end,
            batch_size=args.eval_batch_size,
            device=device,
        )

        validation_metrics = evaluate_nll_metrics(
            model=model,
            arrays=arrays,
            scalers=scalers,
            start=validation_start,
            end=validation_end,
            batch_size=args.eval_batch_size,
            device=device,
        )

        test_metrics, test_predictions, test_avg_probs_movement = evaluate_full(
            model=model,
            arrays=arrays,
            scalers=scalers,
            start=test_start,
            end=test_end,
            batch_size=args.full_eval_batch_size,
            device=device,
            num_classes=num_classes,
        )

        print()
        print(f"Final metrics for checkpoint: {selection_name}")
        print("-" * (31 + len(selection_name)))
        print_metrics("Train", train_metrics)
        print_metrics("Validation", validation_metrics)
        print_metrics("Test", test_metrics)

        print()
        print_most_common_predictions(
            y_pred=test_predictions,
            clip_ticks=clip_ticks,
        )

        return train_metrics, validation_metrics, test_metrics, test_avg_probs_movement

    (
        overall_train_metrics,
        overall_validation_metrics,
        overall_test_metrics,
        overall_test_avg_probs_movement,
    ) = evaluate_checkpoint(
        selection_name="best_validation_nll",
        state_dict=checkpoint_state_dicts["best_validation_nll"],
    )

    (
        movement_train_metrics,
        movement_validation_metrics,
        movement_test_metrics,
        movement_test_avg_probs_movement,
    ) = evaluate_checkpoint(
        selection_name="best_validation_movement_nll",
        state_dict=checkpoint_state_dicts["best_validation_movement_nll"],
    )

    overall_avg_probs_path = (
        results_dir / "full_spatial_nn_best_validation_nll_avg_probs_movement_test.npy"
    )
    movement_avg_probs_path = (
        results_dir
        / "full_spatial_nn_best_validation_movement_nll_avg_probs_movement_test.npy"
    )

    output = {
        "model": "full_local_spatial_neural_network",
        "metadata": metadata,
        "dense_depth": dense_depth,
        "clip_ticks": clip_ticks,
        "num_classes": num_classes,
        "global_depth": args.global_depth,
        "local_window": args.local_window,
        "encoder_hidden_dims": args.encoder_hidden_dims,
        "local_hidden_dims": args.local_hidden_dims,
        "bid_hidden_dims": args.bid_hidden_dims,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "full_eval_batch_size": args.full_eval_batch_size,
        "chunk_size": args.chunk_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "training_time_seconds": float(training_time),
        "checkpoint_info": checkpoint_info,
        "history": history,
        "best_validation_nll_checkpoint": {
            "train_metrics": overall_train_metrics,
            "validation_metrics": overall_validation_metrics,
            "test_metrics": overall_test_metrics,
            "avg_probs_movement_test_file": overall_avg_probs_path.name,
        },
        "best_validation_movement_nll_checkpoint": {
            "train_metrics": movement_train_metrics,
            "validation_metrics": movement_validation_metrics,
            "test_metrics": movement_test_metrics,
            "avg_probs_movement_test_file": movement_avg_probs_path.name,
        },
    }

    output_path = results_dir / "full_spatial_nn.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    np.save(
        overall_avg_probs_path,
        overall_test_avg_probs_movement,
    )

    np.save(
        movement_avg_probs_path,
        movement_test_avg_probs_movement,
    )

    overall_model_path = results_dir / "full_spatial_nn_best_validation_nll.pt"
    movement_model_path = results_dir / "full_spatial_nn_best_validation_movement_nll.pt"

    shared_checkpoint_metadata = {
        "dense_depth": dense_depth,
        "clip_ticks": clip_ticks,
        "global_depth": args.global_depth,
        "local_window": args.local_window,
        "encoder_hidden_dims": args.encoder_hidden_dims,
        "local_hidden_dims": args.local_hidden_dims,
        "bid_hidden_dims": args.bid_hidden_dims,
        "dropout": args.dropout,
        "metadata": metadata,
        "ask_scaler_mean": scalers["ask"].mean_,
        "ask_scaler_scale": scalers["ask"].scale_,
        "bid_scaler_mean": scalers["bid"].mean_,
        "bid_scaler_scale": scalers["bid"].scale_,
        "spread_scaler_mean": scalers["spread"].mean_,
        "spread_scaler_scale": scalers["spread"].scale_,
    }

    torch.save(
        {
            "model_state_dict": checkpoint_state_dicts["best_validation_nll"],
            "selection_metric": "validation_nll",
            "checkpoint_info": checkpoint_info,
            **shared_checkpoint_metadata,
        },
        overall_model_path,
    )

    torch.save(
        {
            "model_state_dict": checkpoint_state_dicts["best_validation_movement_nll"],
            "selection_metric": "validation_movement_nll",
            "checkpoint_info": checkpoint_info,
            **shared_checkpoint_metadata,
        },
        movement_model_path,
    )

    print()
    print(f"Saved results to:                         {output_path}")
    print(f"Saved best-validation-NLL avg probs to:   {overall_avg_probs_path}")
    print(f"Saved best-movement-NLL avg probs to:     {movement_avg_probs_path}")
    print(f"Saved best-validation-NLL model to:       {overall_model_path}")
    print(f"Saved best-validation-movement model to:  {movement_model_path}")


if __name__ == "__main__":
    main()