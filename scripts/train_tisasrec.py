"""Train the basket-aware TiSASRec model on the fixed TemporalRec split.

Run from Jupyter with::

    from scripts.train_tisasrec import run_training
    run_training()

The validation selection metric is exact full-catalogue NDCG@10.  Test data is
never read by this training program.
"""

from __future__ import annotations

import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from temporalrec.data.basket_sequences import (
    TimeAwareNextBasketDataset,
    load_basket_sequence_data,
)
from temporalrec.models.tisasrec import (
    BasketEncoder,
    BasketTiSASRec,
    SampledMultiPositiveLoss,
    TiedItemScorer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "tisasrec_basket_v1"

SEED = 42
MAX_HISTORY_BASKETS = 20
EMBEDDING_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2
FEED_FORWARD_DIM = 128
NUM_TIME_BUCKETS = 32
DROPOUT = 0.10
NUM_NEGATIVE_ITEMS = 256
TRAIN_BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_NORM = 1.0
VALIDATION_K_VALUES = (1, 10, 20, 50)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_full_catalogue_metrics(
    model: BasketTiSASRec,
    loader: DataLoader,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Calculate exact validation ranking metrics without saving predictions."""
    model.eval()
    max_k = max(VALIDATION_K_VALUES)
    totals = {
        k: {"hit_count": 0, "recall_sum": 0.0, "ndcg_sum": 0.0}
        for k in VALIDATION_K_VALUES
    }
    example_count = 0
    discounts = 1.0 / torch.log2(
        torch.arange(2, max_k + 2, device=device, dtype=torch.float32)
    )

    catalogue_vectors = model.basket_encoder.item_embedding.weight[1:].float()
    score_scale = catalogue_vectors.size(1) ** -0.5

    with torch.inference_mode():
        for batch in loader:
            history_items = batch["history_items"].to(device, non_blocking=True)
            history_timestamps = batch["history_timestamps"].to(
                device,
                non_blocking=True,
            )
            target_items = batch["target_items"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                contexts, _, _ = model(history_items, history_timestamps)

            scores = contexts.float() @ catalogue_vectors.T * score_scale
            _, top_positions = torch.topk(scores, k=max_k, dim=1)
            top_item_indices = top_positions + 1

            target_mask = target_items.ne(0)
            relevance = (
                top_item_indices.unsqueeze(2)
                .eq(target_items.unsqueeze(1))
                & target_mask.unsqueeze(1)
            ).any(dim=2)
            target_count = target_mask.sum(dim=1).to(torch.float32)
            example_count += target_items.size(0)

            for k in VALIDATION_K_VALUES:
                relevance_at_k = relevance[:, :k]
                totals[k]["hit_count"] += int(relevance_at_k.any(dim=1).sum().item())
                totals[k]["recall_sum"] += float(
                    (relevance_at_k.sum(dim=1).to(torch.float32) / target_count)
                    .sum()
                    .item()
                )

                dcg = (relevance_at_k.to(torch.float32) * discounts[:k]).sum(dim=1)
                ideal_counts = target_count.to(torch.long).clamp(max=k)
                idcg = torch.zeros_like(dcg)
                for count in range(1, int(target_items.size(1)) + 1):
                    mask = ideal_counts.eq(min(count, k))
                    if mask.any():
                        idcg[mask] = discounts[: min(count, k)].sum()
                totals[k]["ndcg_sum"] += float((dcg / idcg).sum().item())

    return {
        k: {
            "event_hit_rate": totals[k]["hit_count"] / example_count,
            "multi_positive_recall": totals[k]["recall_sum"] / example_count,
            "ndcg": totals[k]["ndcg_sum"] / example_count,
        }
        for k in VALIDATION_K_VALUES
    }


def evaluate_sampled_loss(
    model: BasketTiSASRec,
    loader: DataLoader,
    loss_function: SampledMultiPositiveLoss,
    device: torch.device,
) -> float:
    """Use fixed negative samples so epoch-to-epoch loss remains comparable."""
    model.eval()
    weighted_loss = 0.0
    positive_pairs = 0
    gpu_number = torch.cuda.current_device()

    with torch.inference_mode(), torch.random.fork_rng(devices=[gpu_number]):
        torch.manual_seed(SEED + 10_000)
        torch.cuda.manual_seed_all(SEED + 10_000)
        for batch in loader:
            history_items = batch["history_items"].to(device, non_blocking=True)
            history_timestamps = batch["history_timestamps"].to(
                device,
                non_blocking=True,
            )
            target_items = batch["target_items"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                contexts, _, _ = model(history_items, history_timestamps)
                loss, info = loss_function(contexts, target_items)
            weighted_loss += loss.item() * info["positive_target_pairs"]
            positive_pairs += info["positive_target_pairs"]

    return weighted_loss / positive_pairs


def train_one_epoch(
    model: BasketTiSASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: SampledMultiPositiveLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    weighted_loss = 0.0
    positive_pairs = 0
    start = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        history_items = batch["history_items"].to(device, non_blocking=True)
        history_timestamps = batch["history_timestamps"].to(
            device,
            non_blocking=True,
        )
        target_items = batch["target_items"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            contexts, _, _ = model(history_items, history_timestamps)
            loss, info = loss_function(contexts, target_items)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        weighted_loss += loss.item() * info["positive_target_pairs"]
        positive_pairs += info["positive_target_pairs"]

        if step % 500 == 0:
            print(
                f"  Step {step:,}/{len(loader):,} | "
                f"training loss: {weighted_loss / positive_pairs:.4f}"
            )

    return weighted_loss / positive_pairs, time.perf_counter() - start


def run_training() -> dict[str, Path | float | int]:
    """Train TiSASRec and return locations of its reproducible artifacts."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this configured experiment.")

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_basket_sequence_data(PROJECT_ROOT, MAX_HISTORY_BASKETS)
    if len(data.validation_examples) != 182_936:
        raise AssertionError("unexpected validation example count")

    train_dataset = TimeAwareNextBasketDataset(
        data.train_examples,
        data.basket_item_matrix,
        data.basket_timestamps_ms,
        MAX_HISTORY_BASKETS,
    )
    validation_dataset = TimeAwareNextBasketDataset(
        data.validation_examples,
        data.basket_item_matrix,
        data.basket_timestamps_ms,
        MAX_HISTORY_BASKETS,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    basket_encoder = BasketEncoder(data.num_train_items, EMBEDDING_DIM)
    model = BasketTiSASRec(
        basket_encoder=basket_encoder,
        max_history_baskets=MAX_HISTORY_BASKETS,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        feed_forward_dim=FEED_FORWARD_DIM,
        num_time_buckets=NUM_TIME_BUCKETS,
        dropout=DROPOUT,
    ).to(device)
    scorer = TiedItemScorer(model.basket_encoder.item_embedding, EMBEDDING_DIM)
    loss_function = SampledMultiPositiveLoss(
        scorer,
        data.num_train_items,
        NUM_NEGATIVE_ITEMS,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    checkpoint_path = MODEL_DIR / "best_model_by_validation_ndcg10.pt"
    history_path = MODEL_DIR / "training_history.csv"
    validation_metrics_path = MODEL_DIR / "validation_full_catalogue_metrics_by_epoch.csv"
    history: list[dict[str, float | int | bool]] = []
    validation_rows: list[dict[str, float | int]] = []
    best_ndcg10 = float("-inf")
    epochs_without_improvement = 0

    print(f"Train examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Selection metric: exact full-catalogue validation NDCG@10")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{MAX_EPOCHS}")
        train_loss, epoch_seconds = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
        )
        validation_loss = evaluate_sampled_loss(
            model,
            validation_loader,
            loss_function,
            device,
        )
        validation_metrics = build_full_catalogue_metrics(
            model,
            validation_loader,
            device,
        )
        validation_ndcg10 = validation_metrics[10]["ndcg"]
        improved = validation_ndcg10 > best_ndcg10

        record = {
            "epoch": epoch,
            "train_sampled_loss": train_loss,
            "validation_sampled_loss": validation_loss,
            "validation_ndcg@10": validation_ndcg10,
            "validation_hit_rate@10": validation_metrics[10]["event_hit_rate"],
            "epoch_seconds": epoch_seconds,
            "gpu_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "improved_ndcg@10": improved,
        }
        history.append(record)

        for k, metrics in validation_metrics.items():
            validation_rows.append({"epoch": epoch, "k": k, **metrics})

        print(f"Train sampled loss:      {train_loss:.4f}")
        print(f"Validation sampled loss: {validation_loss:.4f}")
        print(f"Validation NDCG@10:      {validation_ndcg10:.6f}")
        print(f"Epoch time:              {epoch_seconds / 60:.2f} minutes")

        if improved:
            best_ndcg10 = validation_ndcg10
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_ndcg@10": validation_ndcg10,
                    "config": {
                        "seed": SEED,
                        "max_history_baskets": MAX_HISTORY_BASKETS,
                        "max_items_per_basket": data.max_items_per_basket,
                        "num_train_items": data.num_train_items,
                        "embedding_dim": EMBEDDING_DIM,
                        "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS,
                        "feed_forward_dim": FEED_FORWARD_DIM,
                        "num_time_buckets": NUM_TIME_BUCKETS,
                        "time_bucket_rule": "floor(log2(elapsed_seconds + 1)), clipped",
                        "dropout": DROPOUT,
                        "num_negative_items": NUM_NEGATIVE_ITEMS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "training_data": "known_baskets_all_splits.csv",
                        "examples_data": "examples_with_targets.csv",
                        "torch_version": torch.__version__,
                        "pandas_version": pd.__version__,
                        "python_version": platform.python_version(),
                    },
                },
                checkpoint_path,
            )
            print("✅ Best validation NDCG@10 checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(
                "No validation NDCG@10 improvement: "
                f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("⏹️ Early stopping activated.")
                break

        pd.DataFrame(history).to_csv(history_path, index=False)
        pd.DataFrame(validation_rows).to_csv(validation_metrics_path, index=False)

    pd.DataFrame(history).to_csv(history_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_metrics_path, index=False)
    print(f"\n✅ Training finished. Best validation NDCG@10: {best_ndcg10:.6f}")
    print("Checkpoint:", checkpoint_path)

    return {
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "validation_metrics_path": validation_metrics_path,
        "best_validation_ndcg10": best_ndcg10,
        "best_epoch": int(pd.DataFrame(history).loc[
            pd.DataFrame(history)["validation_ndcg@10"].idxmax(), "epoch"
        ]),
    }


if __name__ == "__main__":
    run_training()
