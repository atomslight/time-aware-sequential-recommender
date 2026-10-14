"""Loading and dataset helpers for the prepared basket benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class BasketSequenceData:
    model_baskets: pd.DataFrame
    examples: pd.DataFrame
    basket_item_matrix: np.ndarray
    basket_timestamps_ms: np.ndarray
    train_examples: pd.DataFrame
    validation_examples: pd.DataFrame
    test_examples: pd.DataFrame
    num_train_items: int
    max_items_per_basket: int


def load_basket_sequence_data(
    project_root: Path,
    max_history_baskets: int,
) -> BasketSequenceData:
    """Reload the immutable prepared data and reconstruct numeric basket rows."""
    reframe_dir = (
        project_root
        / "data"
        / "interim"
        / "chronological_split_70_15_15"
        / "training_reframe"
    )

    baskets = pd.read_csv(reframe_dir / "known_baskets_all_splits.csv")
    examples = pd.read_csv(reframe_dir / "examples_with_targets.csv")
    item_mapping = pd.read_csv(reframe_dir / "train_item_id_mapping.csv")

    if not baskets["block_id"].is_unique:
        raise AssertionError("each prepared basket must have one block ID")

    baskets = baskets.sort_values(
        ["visitorid", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    baskets["basket_row_index"] = np.arange(len(baskets), dtype=np.int64)

    visitor_timestamp_gap = baskets.groupby("visitorid", sort=False)["timestamp"].diff()
    if not (visitor_timestamp_gap.dropna() > 0).all():
        raise AssertionError("baskets for one visitor must have strictly increasing timestamps")

    max_items_per_basket = int(baskets["known_item_count"].max())
    basket_item_matrix = np.zeros(
        (len(baskets), max_items_per_basket),
        dtype=np.int64,
    )

    for row_index, item_json in enumerate(baskets["item_indices"]):
        item_indices = json.loads(item_json)
        if not 1 <= len(item_indices) <= max_items_per_basket:
            raise AssertionError("basket membership must be non-empty and bounded")
        basket_item_matrix[row_index, : len(item_indices)] = item_indices

    basket_timestamps_ms = baskets["timestamp"].to_numpy(dtype=np.int64)
    block_to_row = baskets.set_index("block_id")["basket_row_index"]
    examples = examples.copy()
    examples["target_basket_row"] = examples["target_block_id"].map(block_to_row)

    if examples["target_basket_row"].isna().any():
        raise AssertionError("every target block must exist in prepared baskets")
    examples["target_basket_row"] = examples["target_basket_row"].astype(np.int64)

    if not examples["history_length"].between(1, max_history_baskets).all():
        raise AssertionError("history lengths must already be capped and non-empty")

    target_rows = examples["target_basket_row"].to_numpy(dtype=np.int64)
    history_lengths = examples["history_length"].to_numpy(dtype=np.int64)
    history_start_rows = target_rows - history_lengths
    basket_visitors = baskets["visitorid"].to_numpy(dtype=np.int64)

    if not np.array_equal(basket_visitors[target_rows], basket_visitors[history_start_rows]):
        raise AssertionError("history must not cross visitor boundaries")
    if not np.all(basket_timestamps_ms[target_rows] > basket_timestamps_ms[target_rows - 1]):
        raise AssertionError("target baskets must occur after their last history basket")

    num_train_items = int(item_mapping["item_index"].max())
    if basket_item_matrix.max() > num_train_items:
        raise AssertionError("a basket item is outside the training item catalogue")

    return BasketSequenceData(
        model_baskets=baskets,
        examples=examples,
        basket_item_matrix=basket_item_matrix,
        basket_timestamps_ms=basket_timestamps_ms,
        train_examples=examples.loc[examples["split"].eq("train")].reset_index(drop=True),
        validation_examples=examples.loc[examples["split"].eq("validation")].reset_index(drop=True),
        test_examples=examples.loc[examples["split"].eq("test")].reset_index(drop=True),
        num_train_items=num_train_items,
        max_items_per_basket=max_items_per_basket,
    )


class TimeAwareNextBasketDataset(Dataset):
    """Return a right-aligned history, timestamps, and the next basket labels."""

    def __init__(
        self,
        examples: pd.DataFrame,
        basket_item_matrix: np.ndarray,
        basket_timestamps_ms: np.ndarray,
        max_history_baskets: int,
    ) -> None:
        self.basket_item_matrix = basket_item_matrix
        self.basket_timestamps_ms = basket_timestamps_ms
        self.max_history_baskets = max_history_baskets
        self.target_rows = examples["target_basket_row"].to_numpy(dtype=np.int64)
        self.history_lengths = examples["history_length"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.target_rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_row = self.target_rows[index]
        history_length = self.history_lengths[index]
        history_start = target_row - history_length

        history_items = np.zeros(
            (self.max_history_baskets, self.basket_item_matrix.shape[1]),
            dtype=np.int64,
        )
        history_timestamps = np.zeros(self.max_history_baskets, dtype=np.int64)

        history_items[-history_length:] = self.basket_item_matrix[
            history_start:target_row
        ]
        history_timestamps[-history_length:] = self.basket_timestamps_ms[
            history_start:target_row
        ]

        real_timestamps = history_timestamps[-history_length:]
        if not np.all(np.diff(real_timestamps) > 0):
            raise AssertionError("history timestamps must be strictly increasing")
        if not np.all(history_timestamps[:-history_length] == 0):
            raise AssertionError("padding timestamps must remain zero")
        if not real_timestamps[-1] < self.basket_timestamps_ms[target_row]:
            raise AssertionError("target timestamp must be after all history")

        return {
            "history_items": torch.from_numpy(history_items),
            "history_timestamps": torch.from_numpy(history_timestamps),
            "history_length": torch.tensor(history_length, dtype=torch.long),
            "target_items": torch.from_numpy(self.basket_item_matrix[target_row].copy()),
        }
