from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Assumes this file is located in project/src/.
    """
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return get_project_root() / path


# ---------------------------------------------------------------------
# WSELOB reconstruction
# ---------------------------------------------------------------------

def decode_value(value):
    """
    Decode bytes values from HDF5 if needed.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def list_date_keys(h5_path: Path) -> list[str]:
    """
    Return date keys stored in a pandas HDF5 file.
    Example keys look like '/d20170222'.
    """
    with pd.HDFStore(h5_path, mode="r") as store:
        keys = store.keys()

    return sorted(keys)


def ceil_to_interval(timestamp_ns: int, interval_ns: int) -> int:
    """
    Round timestamp up to the next multiple of interval_ns.
    """
    return ((timestamp_ns + interval_ns - 1) // interval_ns) * interval_ns


def update_aggregate_book(
    row,
    bid_book: dict[int, int],
    ask_book: dict[int, int],
) -> None:
    """
    Update aggregate book state using one WSELOB order event.

    We interpret agg_volume as the total aggregate volume at the affected
    price level after the event.
    """
    action_type = decode_value(row.action_type)

    if action_type == "F":
        bid_book.clear()
        ask_book.clear()
        return

    side = int(row.side)
    price = int(row.price)
    agg_volume = int(row.agg_volume)

    if price <= 0:
        return

    if side == 1:
        book = bid_book
    elif side in (2, 5):
        book = ask_book
    else:
        return

    if agg_volume <= 0:
        book.pop(price, None)
    else:
        book[price] = agg_volume


def is_valid_book(bid_book: dict[int, int], ask_book: dict[int, int]) -> bool:
    """
    Check whether the book has both sides and a positive spread.
    """
    if len(bid_book) == 0 or len(ask_book) == 0:
        return False

    best_bid = max(bid_book.keys())
    best_ask = min(ask_book.keys())

    return best_bid < best_ask


def extract_snapshot(
    bid_book: dict[int, int],
    ask_book: dict[int, int],
    timestamp_ns: int,
    date_key: str,
    num_levels: int,
) -> dict:
    """
    Extract a LOBSTER-like snapshot from aggregate bid/ask books.
    """
    bid_levels = sorted(
        [(price, volume) for price, volume in bid_book.items() if volume > 0],
        key=lambda x: x[0],
        reverse=True,
    )[:num_levels]

    ask_levels = sorted(
        [(price, volume) for price, volume in ask_book.items() if volume > 0],
        key=lambda x: x[0],
    )[:num_levels]

    if len(bid_levels) == 0 or len(ask_levels) == 0:
        raise ValueError("Cannot extract snapshot from empty book.")

    snapshot = {
        "date_key": date_key,
        "time": int(timestamp_ns),
    }

    for level in range(num_levels):
        if level < len(ask_levels):
            ask_price, ask_size = ask_levels[level]
        else:
            ask_price, ask_size = 0, 0

        if level < len(bid_levels):
            bid_price, bid_size = bid_levels[level]
        else:
            bid_price, bid_size = 0, 0

        i = level + 1

        snapshot[f"ask_price_{i}"] = int(ask_price)
        snapshot[f"ask_size_{i}"] = int(ask_size)
        snapshot[f"bid_price_{i}"] = int(bid_price)
        snapshot[f"bid_size_{i}"] = int(bid_size)

    snapshot["best_ask"] = snapshot["ask_price_1"]
    snapshot["best_bid"] = snapshot["bid_price_1"]
    snapshot["spread"] = snapshot["best_ask"] - snapshot["best_bid"]

    return snapshot


def reconstruct_snapshots_for_day(
    df: pd.DataFrame,
    date_key: str,
    num_levels: int,
    sample_interval_seconds: float,
) -> pd.DataFrame:
    """
    Reconstruct fixed-interval snapshots from WSELOB event data.

    The book is treated as piecewise constant between events.
    """
    interval_ns = int(sample_interval_seconds * 1_000_000_000)

    df = df.sort_values("time", kind="mergesort")

    bid_book: dict[int, int] = {}
    ask_book: dict[int, int] = {}

    next_sample_time = None
    snapshots = []

    for row in df.itertuples(index=False):
        event_time = int(row.time)

        if next_sample_time is not None:
            while next_sample_time < event_time:
                if is_valid_book(bid_book, ask_book):
                    snapshots.append(
                        extract_snapshot(
                            bid_book=bid_book,
                            ask_book=ask_book,
                            timestamp_ns=next_sample_time,
                            date_key=date_key,
                            num_levels=num_levels,
                        )
                    )

                next_sample_time += interval_ns

        update_aggregate_book(row, bid_book, ask_book)

        if next_sample_time is None and is_valid_book(bid_book, ask_book):
            next_sample_time = ceil_to_interval(event_time, interval_ns)

    return pd.DataFrame(snapshots)


def reconstruct_snapshots_from_file(
    h5_path: Path,
    num_levels: int,
    sample_interval_seconds: float,
    max_days: int | None = None,
) -> pd.DataFrame:
    """
    Reconstruct fixed-interval order book snapshots from all or part of a
    WSELOB HDF5 file.
    """
    keys = list_date_keys(h5_path)

    if max_days is not None:
        keys = keys[:max_days]

    all_snapshots = []

    print("Reconstructing WSELOB snapshots")
    print("===============================")
    print(f"Input file:          {h5_path}")
    print(f"Number of date keys: {len(keys)}")
    print(f"Levels:              {num_levels}")
    print(f"Sampling interval:   {sample_interval_seconds} seconds")
    print()

    for key in keys:
        print(f"Reading {key} ...")

        df = pd.read_hdf(h5_path, key=key)

        snapshots_day = reconstruct_snapshots_for_day(
            df=df,
            date_key=key.strip("/"),
            num_levels=num_levels,
            sample_interval_seconds=sample_interval_seconds,
        )

        print(
            f"  events: {len(df):>8,} | "
            f"snapshots: {len(snapshots_day):>8,}"
        )

        if len(snapshots_day) > 0:
            all_snapshots.append(snapshots_day)

    if len(all_snapshots) == 0:
        raise RuntimeError("No valid snapshots were reconstructed.")

    snapshots = pd.concat(all_snapshots, axis=0, ignore_index=True)

    print()
    print("Snapshot summary")
    print("----------------")
    print(f"Total snapshots:      {len(snapshots):,}")
    print(f"Spread min/max:       {snapshots['spread'].min()} / {snapshots['spread'].max()}")
    print(f"Spread mean:          {snapshots['spread'].mean():.4f}")
    print()

    return snapshots


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def make_feature_and_spatial_arrays(
    snapshots: pd.DataFrame,
    num_levels: int,
    tick_size_raw: int,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], list[str]]:
    """
    Construct the old flat feature matrix and new structured spatial arrays.

    Flat feature order:
        log_ask_sizes        L columns
        log_bid_sizes        L columns
        level_imbalances     L columns
        ask_price_offsets    L columns
        bid_price_offsets    L columns
        spread_ticks         1 column

    Structured spatial arrays are intended for the full spatial neural net.
    """
    ask_size_cols = [f"ask_size_{i}" for i in range(1, num_levels + 1)]
    bid_size_cols = [f"bid_size_{i}" for i in range(1, num_levels + 1)]

    ask_price_cols = [f"ask_price_{i}" for i in range(1, num_levels + 1)]
    bid_price_cols = [f"bid_price_{i}" for i in range(1, num_levels + 1)]

    ask_sizes = snapshots[ask_size_cols].to_numpy(dtype=np.float32)
    bid_sizes = snapshots[bid_size_cols].to_numpy(dtype=np.float32)

    ask_prices = snapshots[ask_price_cols].to_numpy(dtype=np.int64)
    bid_prices = snapshots[bid_price_cols].to_numpy(dtype=np.int64)

    best_ask = snapshots["best_ask"].to_numpy(dtype=np.int64)
    best_bid = snapshots["best_bid"].to_numpy(dtype=np.int64)

    best_ask_float = best_ask.astype(np.float32).reshape(-1, 1)
    best_bid_float = best_bid.astype(np.float32).reshape(-1, 1)

    ask_log_sizes = np.log1p(ask_sizes).astype(np.float32)
    bid_log_sizes = np.log1p(bid_sizes).astype(np.float32)

    denominator = bid_sizes + ask_sizes
    imbalance = np.divide(
        bid_sizes - ask_sizes,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float32),
        where=denominator != 0,
    ).astype(np.float32)

    ask_offsets = (
        (ask_prices.astype(np.float32) - best_ask_float)
        / float(tick_size_raw)
    )

    bid_offsets = (
        (best_bid_float - bid_prices.astype(np.float32))
        / float(tick_size_raw)
    )

    ask_offsets[ask_prices <= 0] = 0.0
    bid_offsets[bid_prices <= 0] = 0.0

    ask_offsets = ask_offsets.astype(np.float32)
    bid_offsets = bid_offsets.astype(np.float32)

    ask_offset_ticks = np.rint(ask_offsets).astype(np.int16)
    bid_offset_ticks = np.rint(bid_offsets).astype(np.int16)

    spread_ticks = (
        snapshots["spread"].to_numpy(dtype=np.float32)
        / float(tick_size_raw)
    ).astype(np.float32)

    feature_parts = [
        ask_log_sizes,
        bid_log_sizes,
        imbalance,
        ask_offsets,
        bid_offsets,
        spread_ticks.reshape(-1, 1),
    ]

    feature_names = []
    feature_names.extend([f"log_ask_size_{i}" for i in range(1, num_levels + 1)])
    feature_names.extend([f"log_bid_size_{i}" for i in range(1, num_levels + 1)])
    feature_names.extend([f"imbalance_{i}" for i in range(1, num_levels + 1)])
    feature_names.extend([f"ask_price_offset_{i}" for i in range(1, num_levels + 1)])
    feature_names.extend([f"bid_price_offset_{i}" for i in range(1, num_levels + 1)])
    feature_names.append("spread_ticks")

    X = np.concatenate(feature_parts, axis=1).astype(np.float32)

    # Encode date keys compactly.
    date_categories = pd.Categorical(snapshots["date_key"])
    date_index = date_categories.codes.astype(np.int16)
    date_keys = [str(x) for x in date_categories.categories.tolist()]

    spatial_arrays = {
        "time": snapshots["time"].to_numpy(dtype=np.int64),
        "date_index": date_index,
        "best_ask": best_ask.astype(np.int32),
        "best_bid": best_bid.astype(np.int32),
        "ask_log_size": ask_log_sizes,
        "bid_log_size": bid_log_sizes,
        "imbalance": imbalance,
        "ask_offset_ticks": ask_offset_ticks,
        "bid_offset_ticks": bid_offset_ticks,
        "spread_ticks": spread_ticks,
    }

    return X, feature_names, spatial_arrays, date_keys


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def make_labels_fixed_horizon(
    snapshots: pd.DataFrame,
    horizon_seconds: float,
    clip_ticks: int,
    tick_size_raw: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct fixed-horizon labels within each date group.

    For each snapshot time t, find the first snapshot at or after t+horizon.
    Labels are clipped ask/bid price changes in tick units.
    """
    horizon_ns = int(horizon_seconds * 1_000_000_000)

    times = snapshots["time"].to_numpy(dtype=np.int64)
    best_ask = snapshots["best_ask"].to_numpy(dtype=np.int64)
    best_bid = snapshots["best_bid"].to_numpy(dtype=np.int64)

    valid_indices = []
    future_indices = []

    for _, positions in snapshots.groupby("date_key", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        group_times = times[positions]

        target_times = group_times + horizon_ns
        future_local_indices = np.searchsorted(
            group_times,
            target_times,
            side="left",
        )

        valid_mask = future_local_indices < len(positions)

        current_positions = positions[valid_mask]
        future_positions = positions[future_local_indices[valid_mask]]

        valid_indices.append(current_positions)
        future_indices.append(future_positions)

    valid_indices = np.concatenate(valid_indices)
    future_indices = np.concatenate(future_indices)

    delta_ask = (
        best_ask[future_indices] - best_ask[valid_indices]
    ) / float(tick_size_raw)

    delta_bid = (
        best_bid[future_indices] - best_bid[valid_indices]
    ) / float(tick_size_raw)

    delta_ask = np.rint(delta_ask).astype(np.int64)
    delta_bid = np.rint(delta_bid).astype(np.int64)

    delta_ask_clipped = np.clip(delta_ask, -clip_ticks, clip_ticks)
    delta_bid_clipped = np.clip(delta_bid, -clip_ticks, clip_ticks)

    y_pair = np.column_stack(
        [delta_ask_clipped, delta_bid_clipped]
    ).astype(np.int64)

    num_values = 2 * clip_ticks + 1

    y_class = (
        (delta_ask_clipped + clip_ticks) * num_values
        + (delta_bid_clipped + clip_ticks)
    ).astype(np.int64)

    return y_pair, y_class, valid_indices


def summarize_labels(y_pair: np.ndarray, clip_ticks: int, top_k: int = 10) -> None:
    """
    Print label distribution summary.
    """
    unique, counts = np.unique(y_pair, axis=0, return_counts=True)
    total = len(y_pair)

    order = np.argsort(counts)[::-1]

    print("Label summary")
    print("-------------")
    print(f"Number of labels:       {total:,}")
    print(f"Clip range:             [-{clip_ticks}, {clip_ticks}]")
    print(f"Number of joint classes: {(2 * clip_ticks + 1) ** 2}")
    print()

    print("Most common labels:")
    for idx in order[:top_k]:
        pair = unique[idx]
        count = counts[idx]
        percentage = 100 * count / total

        print(
            f"  ({pair[0]:>3}, {pair[1]:>3}) : "
            f"{count:>8,} ({percentage:6.2f}%)"
        )

    zero_zero_mask = (y_pair[:, 0] == 0) & (y_pair[:, 1] == 0)

    boundary_mask = (
        (np.abs(y_pair[:, 0]) == clip_ticks)
        | (np.abs(y_pair[:, 1]) == clip_ticks)
    )

    corner_boundary_mask = (
        (np.abs(y_pair[:, 0]) == clip_ticks)
        & (np.abs(y_pair[:, 1]) == clip_ticks)
    )

    print()
    print(f"Proportion of (0, 0):   {100 * zero_zero_mask.mean():.2f}%")
    print(f"Boundary mass:          {100 * boundary_mask.mean():.2f}%")
    print(f"Corner boundary mass:   {100 * corner_boundary_mask.mean():.2f}%")


# ---------------------------------------------------------------------
# Dense tick-grid construction
# ---------------------------------------------------------------------

def make_memmap(path: Path, shape: tuple[int, ...], dtype) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)

    return np.lib.format.open_memmap(
        filename=path,
        mode="w+",
        dtype=dtype,
        shape=shape,
    )


def scatter_log_sizes_to_dense(
    log_size: np.ndarray,
    offset_ticks: np.ndarray,
    dense_max_offset: int,
) -> np.ndarray:
    """
    Convert sparse observed levels into a dense tick-offset grid.

    Output:
        dense[n, k] = log size at tick offset k from best ask/bid.

    Missing tick levels remain zero.
    """
    n_samples, num_levels = log_size.shape

    dense = np.zeros(
        shape=(n_samples, dense_max_offset + 1),
        dtype=np.float32,
    )

    offsets = offset_ticks.astype(np.int64)

    valid = (
        (offsets >= 0)
        & (offsets <= dense_max_offset)
        & np.isfinite(log_size)
        & (log_size > 0.0)
    )

    row_idx, level_idx = np.where(valid)

    dense[
        row_idx,
        offsets[row_idx, level_idx],
    ] = log_size[row_idx, level_idx]

    return dense


def save_dense_spatial_arrays(
    spatial_dir: Path,
    ask_log_size: np.ndarray,
    bid_log_size: np.ndarray,
    ask_offset_ticks: np.ndarray,
    bid_offset_ticks: np.ndarray,
    dense_max_offset: int,
    chunk_size: int,
) -> None:
    """
    Save dense ask/bid log-size arrays in chunks to avoid a large temporary
    allocation.

    Dense arrays have shape:
        (num_samples, dense_max_offset + 1)

    Index k means k ticks away from the current best ask/bid.
    """
    n_samples = ask_log_size.shape[0]

    ask_dense_out = make_memmap(
        spatial_dir / "ask_dense_log_size.npy",
        shape=(n_samples, dense_max_offset + 1),
        dtype=np.float32,
    )

    bid_dense_out = make_memmap(
        spatial_dir / "bid_dense_log_size.npy",
        shape=(n_samples, dense_max_offset + 1),
        dtype=np.float32,
    )

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)

        ask_dense = scatter_log_sizes_to_dense(
            log_size=ask_log_size[start:end],
            offset_ticks=ask_offset_ticks[start:end],
            dense_max_offset=dense_max_offset,
        )

        bid_dense = scatter_log_sizes_to_dense(
            log_size=bid_log_size[start:end],
            offset_ticks=bid_offset_ticks[start:end],
            dense_max_offset=dense_max_offset,
        )

        ask_dense_out[start:end] = ask_dense
        bid_dense_out[start:end] = bid_dense

        print(f"  dense grid rows {start:,} to {end:,} / {n_samples:,}")

    del ask_dense_out
    del bid_dense_out


# ---------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------

def save_dataset(
    output_dir: Path,
    X: np.ndarray,
    y_pair: np.ndarray,
    y_class: np.ndarray,
    feature_names: list[str],
    metadata: dict,
    spatial_arrays: dict[str, np.ndarray],
    date_keys: list[str],
    save_dense_grid: bool,
    dense_max_offset: int,
    dense_chunk_size: int,
) -> None:
    """
    Save processed dataset.

    Old flat outputs are preserved:
        X.npy
        y_pair.npy
        y_class.npy
        feature_names.json
        metadata.json

    New spatial outputs are saved in:
        spatial/
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y_pair.npy", y_pair)
    np.save(output_dir / "y_class.npy", y_class)

    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    spatial_dir = output_dir / "spatial"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Saving spatial arrays")
    print("---------------------")

    for name, array in spatial_arrays.items():
        path = spatial_dir / f"{name}.npy"
        np.save(path, array)
        print(f"  saved {path.name}: shape={array.shape}, dtype={array.dtype}")

    np.save(spatial_dir / "y_pair.npy", y_pair)
    np.save(spatial_dir / "y_class.npy", y_class)

    dense_grid_metadata = {
        "enabled": bool(save_dense_grid),
        "dense_max_offset": int(dense_max_offset) if save_dense_grid else None,
        "ask_dense_log_size": None,
        "bid_dense_log_size": None,
    }

    if save_dense_grid:
        print()
        print("Saving dense tick-grid arrays")
        print("-----------------------------")

        save_dense_spatial_arrays(
            spatial_dir=spatial_dir,
            ask_log_size=spatial_arrays["ask_log_size"],
            bid_log_size=spatial_arrays["bid_log_size"],
            ask_offset_ticks=spatial_arrays["ask_offset_ticks"],
            bid_offset_ticks=spatial_arrays["bid_offset_ticks"],
            dense_max_offset=dense_max_offset,
            chunk_size=dense_chunk_size,
        )

        dense_grid_metadata["ask_dense_log_size"] = "ask_dense_log_size.npy"
        dense_grid_metadata["bid_dense_log_size"] = "bid_dense_log_size.npy"

    spatial_metadata = {
        "description": (
            "Structured arrays for spatial neural network training. "
            "Rows are aligned with X.npy, y_pair.npy and y_class.npy."
        ),
        "arrays": {
            "time": "time.npy",
            "date_index": "date_index.npy",
            "best_ask": "best_ask.npy",
            "best_bid": "best_bid.npy",
            "ask_log_size": "ask_log_size.npy",
            "bid_log_size": "bid_log_size.npy",
            "imbalance": "imbalance.npy",
            "ask_offset_ticks": "ask_offset_ticks.npy",
            "bid_offset_ticks": "bid_offset_ticks.npy",
            "spread_ticks": "spread_ticks.npy",
            "y_pair": "y_pair.npy",
            "y_class": "y_class.npy",
        },
        "date_keys": date_keys,
        "dense_grid": dense_grid_metadata,
        "interpretation": {
            "ask_log_size": "log(1 + ask size) for observed ask levels.",
            "bid_log_size": "log(1 + bid size) for observed bid levels.",
            "ask_offset_ticks": (
                "Tick offset of each observed ask level from current best ask."
            ),
            "bid_offset_ticks": (
                "Tick offset of each observed bid level from current best bid."
            ),
            "dense_log_size": (
                "Dense arrays are indexed by tick offset from current best "
                "ask/bid. Missing levels are zero."
            ),
        },
    }

    with open(spatial_dir / "spatial_metadata.json", "w") as f:
        json.dump(spatial_metadata, f, indent=2)

    metadata = dict(metadata)
    metadata["spatial_arrays"] = {
        "saved": True,
        "directory": "spatial",
        "spatial_metadata": "spatial/spatial_metadata.json",
        "dense_grid": dense_grid_metadata,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Saved processed dataset to: {output_dir}")
    print(f"Saved spatial arrays to:    {spatial_dir}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WSELOB-2017 order data to processed ML arrays."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/WSELOB-2017/orders/KGHM_lob_2017_zlib.h5",
        help="Path to a WSELOB orders .h5 file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/WSELOB_KGHM",
        help="Output directory for processed arrays.",
    )

    parser.add_argument(
        "--num-levels",
        type=int,
        default=50,
        help="Number of bid/ask observed levels to extract.",
    )

    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=1.0,
        help="Snapshot sampling interval in seconds.",
    )

    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=30.0,
        help="Prediction horizon in seconds.",
    )

    parser.add_argument(
        "--clip-ticks",
        type=int,
        default=10,
        help="Clip label movements to [-K, K].",
    )

    parser.add_argument(
        "--tick-size-raw",
        type=int,
        default=5,
        help="Raw price units per tick.",
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=5,
        help="Maximum number of date groups to process. Use -1 for all days.",
    )

    parser.add_argument(
        "--save-dense-grid",
        action="store_true",
        help=(
            "Also save dense tick-grid arrays ask_dense_log_size.npy and "
            "bid_dense_log_size.npy. This can use a lot of disk space."
        ),
    )

    parser.add_argument(
        "--dense-max-offset",
        type=int,
        default=50,
        help=(
            "Maximum tick offset included in dense tick-grid arrays. "
            "Only used when --save-dense-grid is set."
        ),
    )

    parser.add_argument(
        "--dense-chunk-size",
        type=int,
        default=250_000,
        help="Chunk size used when creating dense tick-grid arrays.",
    )

    args = parser.parse_args()

    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output)

    max_days = None if args.max_days == -1 else args.max_days

    snapshots = reconstruct_snapshots_from_file(
        h5_path=input_path,
        num_levels=args.num_levels,
        sample_interval_seconds=args.sample_interval_seconds,
        max_days=max_days,
    )

    X_all, feature_names, spatial_arrays_all, date_keys = (
        make_feature_and_spatial_arrays(
            snapshots=snapshots,
            num_levels=args.num_levels,
            tick_size_raw=args.tick_size_raw,
        )
    )

    y_pair, y_class, valid_indices = make_labels_fixed_horizon(
        snapshots=snapshots,
        horizon_seconds=args.horizon_seconds,
        clip_ticks=args.clip_ticks,
        tick_size_raw=args.tick_size_raw,
    )

    X = X_all[valid_indices]

    spatial_arrays = {
        name: array[valid_indices]
        for name, array in spatial_arrays_all.items()
    }

    metadata = {
        "dataset": "WSELOB-2017",
        "input_file": str(input_path),
        "num_levels_used": int(args.num_levels),
        "sample_interval_seconds": float(args.sample_interval_seconds),
        "horizon_seconds": float(args.horizon_seconds),
        "clip_ticks": int(args.clip_ticks),
        "tick_size_raw": int(args.tick_size_raw),
        "max_days": None if max_days is None else int(max_days),
        "num_snapshots": int(len(snapshots)),
        "num_samples": int(len(X)),
        "num_features": int(X.shape[1]),
        "num_classes": int((2 * args.clip_ticks + 1) ** 2),
        "feature_set": [
            "log_ask_sizes",
            "log_bid_sizes",
            "level_imbalances",
            "relative_price_offsets",
            "spread",
        ],
        "reconstruction_assumption": (
            "agg_volume is interpreted as aggregate volume at the affected "
            "price level after the event."
        ),
        "label_definition": {
            "target": "joint best ask and best bid movement",
            "horizon_seconds": float(args.horizon_seconds),
            "clip_ticks": int(args.clip_ticks),
            "tick_size_raw": int(args.tick_size_raw),
            "class_encoding": (
                "class_id = (delta_ask + K) * (2K + 1) + "
                "(delta_bid + K)"
            ),
        },
    }

    print()
    print("Processed dataset summary")
    print("-------------------------")
    print(f"Snapshots:              {len(snapshots):,}")
    print(f"Usable samples:         {len(X):,}")
    print(f"Features:               {X.shape[1]}")
    print(f"Classes:                {(2 * args.clip_ticks + 1) ** 2}")
    print()

    summarize_labels(y_pair=y_pair, clip_ticks=args.clip_ticks)

    save_dataset(
        output_dir=output_dir,
        X=X,
        y_pair=y_pair,
        y_class=y_class,
        feature_names=feature_names,
        metadata=metadata,
        spatial_arrays=spatial_arrays,
        date_keys=date_keys,
        save_dense_grid=args.save_dense_grid,
        dense_max_offset=args.dense_max_offset,
        dense_chunk_size=args.dense_chunk_size,
    )


if __name__ == "__main__":
    main()