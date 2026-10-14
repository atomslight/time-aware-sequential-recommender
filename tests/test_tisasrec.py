import numpy as np
import pandas as pd
import torch

from temporalrec.data.basket_sequences import TimeAwareNextBasketDataset
from temporalrec.models.tisasrec import (
    BasketEncoder,
    BasketTiSASRec,
    bucket_relative_time_intervals,
)


def test_log_time_buckets_cover_seconds_minutes_and_padding():
    timestamps = torch.tensor([[0, 1_000, 61_000, 86_401_000]])
    valid = torch.tensor([[False, True, True, True]])

    buckets = bucket_relative_time_intervals(timestamps, valid, num_buckets=32)

    # Padding pairs never become usable time features.
    assert buckets[0, 0].eq(0).all()
    # 60 elapsed seconds: floor(log2(60 + 1)) = 5.
    assert buckets[0, 2, 1].item() == 5
    # Roughly one day has a much larger bucket than one minute.
    assert buckets[0, 3, 1].item() > buckets[0, 2, 1].item()


def test_time_aware_dataset_right_aligns_items_and_timestamps():
    examples = pd.DataFrame(
        {"target_basket_row": [2], "history_length": [2]}
    )
    item_matrix = np.array([[1, 0], [2, 3], [4, 0]], dtype=np.int64)
    timestamps = np.array([1_000, 9_000, 20_000], dtype=np.int64)
    dataset = TimeAwareNextBasketDataset(
        examples,
        item_matrix,
        timestamps,
        max_history_baskets=4,
    )

    output = dataset[0]
    assert output["history_items"].shape == (4, 2)
    assert output["history_timestamps"].tolist() == [0, 0, 1_000, 9_000]
    assert output["target_items"].tolist() == [4, 0]


def test_future_basket_cannot_change_earlier_time_aware_encoding():
    torch.manual_seed(7)
    model = BasketTiSASRec(
        basket_encoder=BasketEncoder(num_items=8, embedding_dim=8),
        max_history_baskets=3,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feed_forward_dim=16,
        num_time_buckets=32,
        dropout=0.0,
    ).eval()

    timestamps = torch.tensor([[1_000, 5_000, 15_000]])
    first_history = torch.tensor([[[1], [2], [3]]])
    changed_future = torch.tensor([[[1], [2], [4]]])

    with torch.inference_mode():
        _, encoded_a, _ = model(first_history, timestamps)
        _, encoded_b, _ = model(changed_future, timestamps)

    assert torch.allclose(encoded_a[:, :2], encoded_b[:, :2], atol=1e-6)
