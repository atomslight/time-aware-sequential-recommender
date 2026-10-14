"""Basket-level TiSASRec components.

The source data contains unordered simultaneous-item baskets.  This module
therefore encodes items inside a basket by mean pooling, then applies causal,
time-interval-aware self-attention across baskets.  It never creates an order
among items with the same visitor timestamp.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def bucket_relative_time_intervals(
    timestamps_ms: Tensor,
    valid_mask: Tensor,
    num_buckets: int,
) -> Tensor:
    """Return log-bucketed elapsed-time intervals for every query/key pair.

    ``timestamps_ms`` is right-aligned history time with zero padding.  Under
    the causal attention mask, a query can only attend to itself or an earlier
    key, so ``query_time - key_time`` is a non-negative elapsed interval.
    Padding entries are assigned bucket zero and are later masked from attention.
    """
    if timestamps_ms.ndim != 2:
        raise ValueError("timestamps_ms must have shape [batch, sequence]")
    if valid_mask.shape != timestamps_ms.shape:
        raise ValueError("valid_mask must match timestamps_ms")
    if num_buckets < 2:
        raise ValueError("num_buckets must be at least 2")

    query_time = timestamps_ms.unsqueeze(2)
    key_time = timestamps_ms.unsqueeze(1)
    elapsed_ms = (query_time - key_time).clamp_min(0)

    elapsed_seconds = elapsed_ms.to(torch.float32) / 1_000.0
    buckets = torch.floor(torch.log2(elapsed_seconds + 1.0)).to(torch.long)
    buckets = buckets.clamp(max=num_buckets - 1)

    pair_is_real = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    return buckets.masked_fill(~pair_is_real, 0)


class BasketEncoder(nn.Module):
    """Mean-pool item embeddings inside one unordered timestamp basket."""

    def __init__(self, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

    def forward(self, basket_item_ids: Tensor) -> tuple[Tensor, Tensor]:
        item_mask = basket_item_ids.ne(0)
        item_vectors = self.item_embedding(basket_item_ids)

        item_count = item_mask.sum(dim=2, keepdim=True).clamp_min(1)
        basket_vectors = (
            item_vectors * item_mask.unsqueeze(-1)
        ).sum(dim=2) / item_count

        basket_mask = item_mask.any(dim=2)
        basket_vectors = basket_vectors * basket_mask.unsqueeze(-1)
        return basket_vectors, basket_mask


class TimeAwareMultiHeadAttention(nn.Module):
    """Causal self-attention with learned relative-time key/value vectors."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_time_buckets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must divide evenly by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # TiSASRec-style relative-time additions for both keys and values.
        self.time_key_embedding = nn.Embedding(
            num_time_buckets,
            embedding_dim,
        )
        self.time_value_embedding = nn.Embedding(
            num_time_buckets,
            embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, sequence_length, _ = values.shape
        return values.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def forward(
        self,
        hidden: Tensor,
        relative_time_buckets: Tensor,
        valid_mask: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden.shape

        query = self._split_heads(self.query(hidden))
        key = self._split_heads(self.key(hidden))
        value = self._split_heads(self.value(hidden))

        relative_key = self.time_key_embedding(relative_time_buckets)
        relative_value = self.time_value_embedding(relative_time_buckets)

        relative_key = relative_key.view(
            batch_size,
            sequence_length,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 3, 1, 2, 4)

        relative_value = relative_value.view(
            batch_size,
            sequence_length,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 3, 1, 2, 4)

        key_with_time = key.unsqueeze(2) + relative_key
        attention_scores = (
            query.unsqueeze(3) * key_with_time
        ).sum(dim=-1) * self.scale

        allowed = (
            ~causal_mask.unsqueeze(0).unsqueeze(0)
        ) & valid_mask.unsqueeze(1).unsqueeze(3) & valid_mask.unsqueeze(1).unsqueeze(2)

        # Zeroing after softmax prevents NaNs for fully padded query rows.
        attention_scores = attention_scores.masked_fill(
            ~allowed,
            -10_000.0,
        )
        attention = torch.softmax(attention_scores, dim=-1)
        attention = attention * allowed.to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        attention = self.dropout(attention)

        value_with_time = value.unsqueeze(2) + relative_value
        attended = (attention.unsqueeze(-1) * value_with_time).sum(dim=3)

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )
        return self.output(attended) * valid_mask.unsqueeze(-1)


class TimeAwareTransformerBlock(nn.Module):
    """A compact pre-norm TiSASRec encoder block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feed_forward_dim: int,
        num_time_buckets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = TimeAwareMultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_time_buckets=num_time_buckets,
            dropout=dropout,
        )
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        relative_time_buckets: Tensor,
        valid_mask: Tensor,
        causal_mask: Tensor,
    ) -> Tensor:
        attention_output = self.attention(
            self.attention_norm(hidden),
            relative_time_buckets,
            valid_mask,
            causal_mask,
        )
        hidden = hidden + self.dropout(attention_output)
        hidden = hidden * valid_mask.unsqueeze(-1)

        feed_forward_output = self.feed_forward(
            self.feed_forward_norm(hidden)
        )
        hidden = hidden + self.dropout(feed_forward_output)
        return hidden * valid_mask.unsqueeze(-1)


class BasketTiSASRec(nn.Module):
    """Time-interval-aware next-basket recommendation model."""

    def __init__(
        self,
        basket_encoder: BasketEncoder,
        max_history_baskets: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dim: int,
        num_time_buckets: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.basket_encoder = basket_encoder
        self.max_history_baskets = max_history_baskets
        self.num_time_buckets = num_time_buckets

        self.position_embedding = nn.Embedding(
            max_history_baskets,
            embedding_dim,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TimeAwareTransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    feed_forward_dim=feed_forward_dim,
                    num_time_buckets=num_time_buckets,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(embedding_dim)

        causal_mask = torch.triu(
            torch.ones(
                max_history_baskets,
                max_history_baskets,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        history_items: Tensor,
        history_timestamps_ms: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        basket_vectors, basket_mask = self.basket_encoder(history_items)
        _, sequence_length, _ = basket_vectors.shape

        if sequence_length != self.max_history_baskets:
            raise ValueError("unexpected history sequence length")
        if history_timestamps_ms.shape != basket_mask.shape:
            raise ValueError("history timestamps must align with basket history")

        positions = torch.arange(
            sequence_length,
            device=history_items.device,
        )
        hidden = basket_vectors + self.position_embedding(positions).unsqueeze(0)
        hidden = self.input_dropout(hidden * basket_mask.unsqueeze(-1))

        relative_time_buckets = bucket_relative_time_intervals(
            history_timestamps_ms,
            basket_mask,
            self.num_time_buckets,
        )

        for block in self.blocks:
            hidden = block(
                hidden,
                relative_time_buckets,
                basket_mask,
                self.causal_mask,
            )

        encoded_history = self.output_norm(hidden) * basket_mask.unsqueeze(-1)

        positions = torch.arange(
            sequence_length,
            device=history_items.device,
        ).unsqueeze(0)
        last_real_position = torch.where(
            basket_mask,
            positions,
            torch.full_like(positions, -1),
        ).max(dim=1).values

        if not torch.all(last_real_position >= 0):
            raise ValueError("every example needs at least one history basket")

        latest_context = encoded_history[
            torch.arange(history_items.size(0), device=history_items.device),
            last_real_position,
        ]
        return latest_context, encoded_history, basket_mask


class TiedItemScorer(nn.Module):
    """Score candidates with the basket encoder's same item embedding table."""

    def __init__(self, item_embedding: nn.Embedding, embedding_dim: int) -> None:
        super().__init__()
        self.item_embedding = item_embedding
        self.score_scale = embedding_dim**-0.5

    def forward(self, context_vectors: Tensor, candidate_item_ids: Tensor) -> Tensor:
        candidate_vectors = self.item_embedding(candidate_item_ids)
        return (
            context_vectors.unsqueeze(1) * candidate_vectors
        ).sum(dim=-1) * self.score_scale


class SampledMultiPositiveLoss(nn.Module):
    """Sampled softmax where every next-basket item is a positive label."""

    def __init__(
        self,
        scorer: TiedItemScorer,
        num_items: int,
        num_negative_items: int,
    ) -> None:
        super().__init__()
        self.scorer = scorer
        self.num_items = num_items
        self.num_negative_items = num_negative_items

    def forward(self, contexts: Tensor, target_items: Tensor) -> tuple[Tensor, dict[str, int]]:
        target_mask = target_items.ne(0)
        example_indices, _ = target_mask.nonzero(as_tuple=True)
        positive_items = target_items[target_mask]

        if positive_items.numel() == 0:
            raise ValueError("every batch must contain a positive target item")

        positive_contexts = contexts[example_indices]
        positive_count = positive_items.numel()

        negatives = torch.randint(
            low=1,
            high=self.num_items + 1,
            size=(positive_count, self.num_negative_items),
            device=contexts.device,
        )

        # A negative cannot be any known target from its own basket.
        positive_sets = target_items[example_indices]
        conflict = (
            negatives.unsqueeze(-1).eq(positive_sets.unsqueeze(1))
            & positive_sets.unsqueeze(1).ne(0)
        ).any(dim=-1)

        while conflict.any():
            negatives[conflict] = torch.randint(
                low=1,
                high=self.num_items + 1,
                size=(int(conflict.sum().item()),),
                device=contexts.device,
            )
            conflict = (
                negatives.unsqueeze(-1).eq(positive_sets.unsqueeze(1))
                & positive_sets.unsqueeze(1).ne(0)
            ).any(dim=-1)

        candidates = torch.cat(
            [positive_items.unsqueeze(1), negatives],
            dim=1,
        )
        scores = self.scorer(positive_contexts, candidates)
        labels = torch.zeros(positive_count, dtype=torch.long, device=contexts.device)
        loss = F.cross_entropy(scores, labels)
        return loss, {"positive_target_pairs": int(positive_count)}
