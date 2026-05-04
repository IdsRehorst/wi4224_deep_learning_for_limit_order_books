from pathlib import Path
import pandas as pd


MESSAGE_COLUMNS = [
    "time",
    "event_type",
    "order_id",
    "size",
    "price",
    "direction",
]


def make_orderbook_columns(num_levels: int) -> list[str]:
    """
    LOBSTER orderbook format:
    ask_price_1, ask_size_1, bid_price_1, bid_size_1,
    ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...
    """
    columns = []

    for level in range(1, num_levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )

    return columns


def read_message_file(message_path: str | Path) -> pd.DataFrame:
    """
    Read a LOBSTER message file.

    The time column is measured in seconds after midnight.
    Prices are usually stored as integer prices multiplied by 10000.
    """
    message_path = Path(message_path)

    messages = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
    )

    return messages


def read_orderbook_file(
    orderbook_path: str | Path,
    max_levels: int | None = None,
) -> pd.DataFrame:
    """
    Read a LOBSTER orderbook file.

    If max_levels is provided, only the first max_levels levels are retained.
    """
    orderbook_path = Path(orderbook_path)

    raw = pd.read_csv(orderbook_path, header=None)

    if raw.shape[1] % 4 != 0:
        raise ValueError(
            f"Orderbook file has {raw.shape[1]} columns, which is not divisible by 4."
        )

    num_levels_available = raw.shape[1] // 4

    if max_levels is None:
        num_levels = num_levels_available
    else:
        if max_levels > num_levels_available:
            raise ValueError(
                f"Requested {max_levels} levels, but file only contains "
                f"{num_levels_available} levels."
            )
        num_levels = max_levels
        raw = raw.iloc[:, : 4 * max_levels]

    raw.columns = make_orderbook_columns(num_levels)

    return raw


def load_lobster_data(
    message_path: str | Path,
    orderbook_path: str | Path,
    max_levels: int | None = None,
) -> pd.DataFrame:
    """
    Load and combine LOBSTER message and orderbook files.

    Each row of the orderbook file corresponds to the state of the book
    after the event in the corresponding row of the message file.
    """
    messages = read_message_file(message_path)
    orderbook = read_orderbook_file(orderbook_path, max_levels=max_levels)

    if len(messages) != len(orderbook):
        raise ValueError(
            f"Message and orderbook files have different lengths: "
            f"{len(messages)} messages vs {len(orderbook)} orderbook rows."
        )

    derived = pd.DataFrame(
        {
            "best_ask": orderbook["ask_price_1"],
            "best_bid": orderbook["bid_price_1"],
            "spread": orderbook["ask_price_1"] - orderbook["bid_price_1"],
        }
    )

    data = pd.concat([messages, orderbook, derived], axis=1)

    return data


def summarize_lobster_data(data: pd.DataFrame) -> None:
    """
    Print basic sanity checks for a loaded LOBSTER dataframe.
    """
    price_columns = [
        col for col in data.columns
        if col.startswith("ask_price_") or col.startswith("bid_price_")
    ]

    num_levels = len(price_columns) // 2

    print("LOBSTER data summary")
    print("--------------------")
    print(f"Number of rows:        {len(data):,}")
    print(f"Number of levels:      {num_levels}")
    print(f"First timestamp:       {data['time'].iloc[0]:.6f}")
    print(f"Last timestamp:        {data['time'].iloc[-1]:.6f}")
    print(f"Best ask min/max:      {data['best_ask'].min()} / {data['best_ask'].max()}")
    print(f"Best bid min/max:      {data['best_bid'].min()} / {data['best_bid'].max()}")
    print(f"Spread min/max:        {data['spread'].min()} / {data['spread'].max()}")
    print(f"Spread mean:           {data['spread'].mean():.2f}")
    print()
    print("First rows:")
    print(data.head())


if __name__ == "__main__":
    sample_dir = Path("../data/raw/LOBSTER_SampleFile_AAPL_2012-06-21_50")

    message_path = sample_dir / "AAPL_2012-06-21_34200000_37800000_message_50.csv"
    orderbook_path = sample_dir / "AAPL_2012-06-21_34200000_37800000_orderbook_50.csv"

    data = load_lobster_data(
        message_path=message_path,
        orderbook_path=orderbook_path,
        max_levels=50,
    )

    summarize_lobster_data(data)