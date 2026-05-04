from pathlib import Path
import json

import numpy as np
import pandas as pd

from load_lobster import load_lobster_data


TICK_SIZE_RAW = 100  # LOBSTER prices are multiplied by 10000, so $0.01 = 100


def get_project_root() -> Path:
    """
    Assumes this file is located in project/src/.
    """
    return Path(__file__).resolve().parents[1]


def find_lobster_files(sample_dir: Path) -> tuple[Path, Path]:
    """
    Find the message and orderbook files inside a LOBSTER sample directory.
    """
    message_files = list(sample_dir.glob("*message*.csv"))
    orderbook_files = list(sample_dir.glob("*orderbook*.csv"))

    if len(message_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one message file in {sample_dir}, "
            f"but found {len(message_files)}."
        )

    if len(orderbook_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one orderbook file in {sample_dir}, "
            f"but found {len(orderbook_files)}."
        )

    return message_files[0], orderbook_files[0]


def make_feature_matrix(
    data: pd.DataFrame,
    num_levels: int,
    add_imbalance: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Construct feature matrix X from the limit order book state.

    First version:
    - ask sizes for levels 1,...,L
    - bid sizes for levels 1,...,L
    - optionally level-wise order book imbalances
    """
    feature_parts = []
    feature_names = []

    ask_size_cols = [f"ask_size_{i}" for i in range(1, num_levels + 1)]
    bid_size_cols = [f"bid_size_{i}" for i in range(1, num_levels + 1)]

    ask_sizes = data[ask_size_cols].to_numpy(dtype=np.float32)
    bid_sizes = data[bid_size_cols].to_numpy(dtype=np.float32)

    # log-transform volumes to reduce the effect of very large orders
    ask_features = np.log1p(ask_sizes)
    bid_features = np.log1p(bid_sizes)

    feature_parts.extend([ask_features, bid_features])
    feature_names.extend([f"log_ask_size_{i}" for i in range(1, num_levels + 1)])
    feature_names.extend([f"log_bid_size_{i}" for i in range(1, num_levels + 1)])

    if add_imbalance:
        denominator = bid_sizes + ask_sizes
        imbalance = np.divide(
            bid_sizes - ask_sizes,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float32),
            where=denominator != 0,
        )

        feature_parts.append(imbalance)
        feature_names.extend([f"imbalance_{i}" for i in range(1, num_levels + 1)])

    X = np.concatenate(feature_parts, axis=1)

    return X.astype(np.float32), feature_names


def make_labels_fixed_horizon(
    data: pd.DataFrame,
    horizon_seconds: float,
    clip_ticks: int,
    tick_size_raw: int = TICK_SIZE_RAW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct labels for the fixed-horizon prediction problem.

    For each event time t, find the first event at or after t + horizon_seconds.
    Then compute:

        Delta A_t = (A_{t+h} - A_t) / tick_size
        Delta B_t = (B_{t+h} - B_t) / tick_size

    The movements are clipped to [-clip_ticks, clip_ticks].

    Returns:
    - y_pair: shape (N, 2), clipped ask and bid movements
    - y_class: shape (N,), class index for the joint movement
    - valid_indices: original row indices used as current times
    """
    times = data["time"].to_numpy(dtype=np.float64)
    best_ask = data["best_ask"].to_numpy(dtype=np.int64)
    best_bid = data["best_bid"].to_numpy(dtype=np.int64)

    future_times = times + horizon_seconds
    future_indices = np.searchsorted(times, future_times, side="left")

    valid_mask = future_indices < len(data)
    valid_indices = np.where(valid_mask)[0]
    future_indices = future_indices[valid_mask]

    delta_ask = (best_ask[future_indices] - best_ask[valid_indices]) / tick_size_raw
    delta_bid = (best_bid[future_indices] - best_bid[valid_indices]) / tick_size_raw

    delta_ask = np.rint(delta_ask).astype(np.int64)
    delta_bid = np.rint(delta_bid).astype(np.int64)

    delta_ask_clipped = np.clip(delta_ask, -clip_ticks, clip_ticks)
    delta_bid_clipped = np.clip(delta_bid, -clip_ticks, clip_ticks)

    y_pair = np.column_stack([delta_ask_clipped, delta_bid_clipped]).astype(np.int64)

    # Map pairs in {-K,...,K}^2 to class labels 0,...,(2K+1)^2-1.
    num_values = 2 * clip_ticks + 1
    y_class = (
        (delta_ask_clipped + clip_ticks) * num_values
        + (delta_bid_clipped + clip_ticks)
    ).astype(np.int64)

    return y_pair, y_class, valid_indices


def summarize_labels(y_pair: np.ndarray, clip_ticks: int) -> None:
    """
    Print a compact summary of the joint label distribution.
    """
    unique, counts = np.unique(y_pair, axis=0, return_counts=True)
    total = len(y_pair)

    order = np.argsort(counts)[::-1]

    print("Label summary")
    print("-------------")
    print(f"Number of labels:       {total:,}")
    print(f"Clip range:             [-{clip_ticks}, {clip_ticks}] ticks")
    print(f"Number of joint classes: {(2 * clip_ticks + 1) ** 2}")
    print()

    print("Most common labels:")
    for idx in order[:10]:
        pair = unique[idx]
        count = counts[idx]
        percentage = 100 * count / total
        print(f"  ({pair[0]:>2}, {pair[1]:>2}) : {count:>8,}  ({percentage:5.2f}%)")

    zero_zero_mask = (y_pair[:, 0] == 0) & (y_pair[:, 1] == 0)
    print()
    print(f"Proportion of (0, 0):   {100 * zero_zero_mask.mean():.2f}%")


def save_dataset(
    output_dir: Path,
    X: np.ndarray,
    y_pair: np.ndarray,
    y_class: np.ndarray,
    feature_names: list[str],
    metadata: dict,
) -> None:
    """
    Save processed arrays and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y_pair.npy", y_pair)
    np.save(output_dir / "y_class.npy", y_class)

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved processed dataset to: {output_dir}")


def main() -> None:
    project_root = get_project_root()

    sample_dir = (
        project_root
        / "data"
        / "raw"
        / "LOBSTER_SampleFile_AAPL_2012-06-21_50"
    )

    output_dir = project_root / "data" / "processed" / "AAPL_2012-06-21_50"

    num_levels = 10
    horizon_seconds = 1.0
    clip_ticks = 10
    add_imbalance = True

    message_path, orderbook_path = find_lobster_files(sample_dir)

    print(f"Using message file:   {message_path}")
    print(f"Using orderbook file: {orderbook_path}")
    print()

    data = load_lobster_data(
        message_path=message_path,
        orderbook_path=orderbook_path,
        max_levels=50,
    )

    X_all, feature_names = make_feature_matrix(
        data=data,
        num_levels=num_levels,
        add_imbalance=add_imbalance,
    )

    y_pair, y_class, valid_indices = make_labels_fixed_horizon(
        data=data,
        horizon_seconds=horizon_seconds,
        clip_ticks=clip_ticks,
    )

    X = X_all[valid_indices]

    metadata = {
        "ticker": "AAPL",
        "date": "2012-06-21",
        "num_levels_used": num_levels,
        "horizon_seconds": horizon_seconds,
        "clip_ticks": clip_ticks,
        "tick_size_raw": TICK_SIZE_RAW,
        "add_imbalance": add_imbalance,
        "num_raw_rows": int(len(data)),
        "num_samples": int(len(X)),
        "num_features": int(X.shape[1]),
        "num_classes": int((2 * clip_ticks + 1) ** 2),
    }

    print("Dataset summary")
    print("---------------")
    print(f"Raw rows:              {len(data):,}")
    print(f"Usable samples:        {len(X):,}")
    print(f"Features:              {X.shape[1]}")
    print(f"Prediction horizon:    {horizon_seconds} seconds")
    print(f"Clip ticks:            {clip_ticks}")
    print()

    summarize_labels(y_pair, clip_ticks=clip_ticks)

    save_dataset(
        output_dir=output_dir,
        X=X,
        y_pair=y_pair,
        y_class=y_class,
        feature_names=feature_names,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()