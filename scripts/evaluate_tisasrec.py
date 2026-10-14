"""Full-catalogue TiSASRec evaluation and fair baseline comparison.

This script performs no optimisation.  It loads the validation-selected
checkpoint once, scores every held-out test history against the training item
catalogue, and writes auditable per-example results.
"""

from __future__ import annotations

import json
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
from temporalrec.models.tisasrec import BasketEncoder, BasketTiSASRec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "tisasrec_basket_v1"
SASREC_MODEL_DIR = PROJECT_ROOT / "models" / "sasrec_basket_v1"
REFRAME_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "chronological_split_70_15_15"
    / "training_reframe"
)

MAX_HISTORY_BASKETS = 20
EVALUATION_BATCH_SIZE = 256
K_VALUES = (1, 10, 20, 50)
MAX_K = max(K_VALUES)
SAVE_TOP_50_ITEM_IDS = False


def load_item_lookup(num_items: int) -> np.ndarray:
    mapping = pd.read_csv(REFRAME_DIR / "train_item_id_mapping.csv")
    lookup = np.zeros(num_items + 1, dtype=np.int64)
    lookup[
        mapping["item_index"].to_numpy(dtype=np.int64)
    ] = mapping["itemid"].to_numpy(dtype=np.int64)
    if not (lookup[1:] != 0).all():
        raise AssertionError("every model item index must map to one source item ID")
    return lookup


def metric_state() -> dict[int, dict[str, float]]:
    return {
        k: {"hit_count": 0.0, "recall_sum": 0.0, "ndcg_sum": 0.0}
        for k in K_VALUES
    }


def metrics_frame(
    totals: dict[int, dict[str, float]],
    example_count: int,
    model_name: str,
    checkpoint_epoch: int | None,
    elapsed_seconds: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "evaluation_protocol": "full catalogue: every training-known item scored for every test example",
                "checkpoint_epoch": checkpoint_epoch,
                "test_examples": example_count,
                "catalogue_items": None,  # filled by caller for clear CSV reading
                "k": k,
                "event_hit_rate": totals[k]["hit_count"] / example_count,
                "multi_positive_recall": totals[k]["recall_sum"] / example_count,
                "ndcg": totals[k]["ndcg_sum"] / example_count,
                "elapsed_seconds": round(elapsed_seconds, 2),
            }
            for k in K_VALUES
        ]
    )


def update_totals(
    totals: dict[int, dict[str, float]],
    relevance_top_50: torch.Tensor,
    target_counts: torch.Tensor,
    discounts: torch.Tensor,
) -> None:
    for k in K_VALUES:
        relevance = relevance_top_50[:, :k]
        totals[k]["hit_count"] += float(relevance.any(dim=1).sum().item())
        totals[k]["recall_sum"] += float(
            (relevance.sum(dim=1).to(torch.float32) / target_counts).sum().item()
        )
        dcg = (relevance.to(torch.float32) * discounts[:k]).sum(dim=1)
        ideal_counts = target_counts.to(torch.long).clamp(max=k)
        idcg = torch.zeros_like(dcg)
        max_targets = relevance_top_50.size(2) if relevance_top_50.ndim == 3 else 6
        # The target count is never zero; basket size is at most six.
        for count in range(1, max_targets + 1):
            mask = ideal_counts.eq(min(count, k))
            if mask.any():
                idcg[mask] = discounts[: min(count, k)].sum()
        totals[k]["ndcg_sum"] += float((dcg / idcg).sum().item())


def write_tisasrec_audit(
    model: BasketTiSASRec,
    loader: DataLoader,
    test_examples: pd.DataFrame,
    index_to_itemid: np.ndarray,
    checkpoint_epoch: int,
    device: torch.device,
) -> tuple[pd.DataFrame, Path, Path, Path]:
    """Score all test contexts and save compact vectors plus a one-row audit."""
    model.eval()
    output_path = MODEL_DIR / "tisasrec_full_catalogue_test_per_example.csv"
    metrics_path = MODEL_DIR / "tisasrec_full_catalogue_test_metrics.csv"
    context_path = MODEL_DIR / "tisasrec_test_context_vectors.npy"
    metadata_path = MODEL_DIR / "tisasrec_test_context_metadata.csv"

    if output_path.exists():
        output_path.unlink()

    context_store = np.lib.format.open_memmap(
        context_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(test_examples), model.basket_encoder.item_embedding.embedding_dim),
    )

    totals = metric_state()
    discounts = 1.0 / torch.log2(
        torch.arange(2, MAX_K + 2, device=device, dtype=torch.float32)
    )
    item_vectors = model.basket_encoder.item_embedding.weight[1:].float()
    score_scale = item_vectors.size(1) ** -0.5
    start = time.perf_counter()
    next_row = 0

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            history_items = batch["history_items"].to(device, non_blocking=True)
            history_timestamps = batch["history_timestamps"].to(
                device,
                non_blocking=True,
            )
            target_items = batch["target_items"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                contexts, _, _ = model(history_items, history_timestamps)

            contexts_float = contexts.float()
            scores = contexts_float @ item_vectors.T * score_scale
            top_scores, top_positions = torch.topk(scores, k=MAX_K, dim=1)
            top_item_indices = top_positions + 1

            target_mask = target_items.ne(0)
            target_counts = target_mask.sum(dim=1).to(torch.float32)
            relevance = (
                top_item_indices.unsqueeze(2)
                .eq(target_items.unsqueeze(1))
                & target_mask.unsqueeze(1)
            ).any(dim=2)
            update_totals(totals, relevance, target_counts, discounts)

            safe_target_positions = target_items.clamp_min(1) - 1
            target_scores = scores.gather(1, safe_target_positions).masked_fill(
                ~target_mask,
                float("-inf"),
            )
            best_actual_scores, best_actual_slots = target_scores.max(dim=1)
            best_actual_indices = target_items.gather(
                1,
                best_actual_slots.unsqueeze(1),
            ).squeeze(1)
            best_actual_ranks = scores.gt(best_actual_scores.unsqueeze(1)).sum(dim=1) + 1

            batch_size = target_items.size(0)
            context_store[next_row : next_row + batch_size] = contexts_float.cpu().numpy()

            top_indices_np = top_item_indices.cpu().numpy()
            top_scores_np = top_scores.cpu().numpy()
            relevance_np = relevance.cpu().numpy()
            targets_np = target_items.cpu().numpy()
            best_indices_np = best_actual_indices.cpu().numpy()
            best_scores_np = best_actual_scores.cpu().numpy()
            best_ranks_np = best_actual_ranks.cpu().numpy()
            metadata = test_examples.iloc[next_row : next_row + batch_size]
            output_rows = []

            for local_row, (_, row) in enumerate(metadata.iterrows()):
                actual_indices = targets_np[local_row][targets_np[local_row] != 0]
                actual_itemids = [int(index_to_itemid[value]) for value in actual_indices]
                top_itemids = index_to_itemid[top_indices_np[local_row]]
                retrieved_actual = [
                    int(itemid)
                    for itemid, relevant in zip(top_itemids, relevance_np[local_row])
                    if relevant
                ]
                result = {
                    "context_row_index": next_row + local_row,
                    "visitorid": int(row["visitorid"]),
                    "target_timestamp": int(row["target_timestamp"]),
                    "target_block_id": row["target_block_id"],
                    "history_length": int(row["history_length"]),
                    "actual_next_itemids": json.dumps(actual_itemids),
                    "actual_item_count": len(actual_itemids),
                    "predicted_top_1_itemid": int(top_itemids[0]),
                    "predicted_top_1_logit_score": round(float(top_scores_np[local_row, 0]), 6),
                    "top_1_result": "TP" if relevance_np[local_row, 0] else "MISS",
                    "top_50_result": "TP_IN_TOP_50" if relevance_np[local_row].any() else "FAILED_TOP_50",
                    "hit_at_10": bool(relevance_np[local_row, :10].any()),
                    "hit_at_20": bool(relevance_np[local_row, :20].any()),
                    "hit_at_50": bool(relevance_np[local_row].any()),
                    "best_actual_itemid": int(index_to_itemid[best_indices_np[local_row]]),
                    "best_actual_logit_score": round(float(best_scores_np[local_row]), 6),
                    "best_actual_full_catalogue_rank": int(best_ranks_np[local_row]),
                    "retrieved_actual_itemids_in_top_50": json.dumps(retrieved_actual),
                    "retrieved_actual_item_count_in_top_50": len(retrieved_actual),
                }
                if SAVE_TOP_50_ITEM_IDS:
                    result["top_50_predicted_itemids"] = json.dumps(
                        [int(itemid) for itemid in top_itemids]
                    )
                output_rows.append(result)

            pd.DataFrame(output_rows).to_csv(
                output_path,
                mode="w" if next_row == 0 else "a",
                header=next_row == 0,
                index=False,
            )
            next_row += batch_size

            if batch_number % 20 == 0 or next_row == len(test_examples):
                print(f"  Scored {next_row:,}/{len(test_examples):,} test examples")

    context_store.flush()
    if next_row != len(test_examples):
        raise AssertionError("every test example must receive exactly one prediction")

    context_metadata = test_examples[
        [
            "visitorid",
            "target_timestamp",
            "target_block_id",
            "history_length",
            "target_item_indices",
            "target_item_count",
        ]
    ].copy()
    context_metadata.insert(0, "context_row_index", np.arange(len(context_metadata)))
    context_metadata.to_csv(metadata_path, index=False)

    elapsed = time.perf_counter() - start
    metrics = metrics_frame(totals, next_row, "tisasrec", checkpoint_epoch, elapsed)
    metrics["catalogue_items"] = len(index_to_itemid) - 1
    metrics.to_csv(metrics_path, index=False)

    # The aggregate file must reconcile with the per-example audit.
    audit_counts = pd.read_csv(output_path, usecols=["top_1_result", "top_50_result"])
    if int(audit_counts["top_1_result"].eq("TP").sum()) != int(totals[1]["hit_count"]):
        raise AssertionError("Top-1 audit count does not match aggregate metric")
    if int(audit_counts["top_50_result"].eq("TP_IN_TOP_50").sum()) != int(totals[50]["hit_count"]):
        raise AssertionError("Top-50 audit count does not match aggregate metric")

    return metrics, output_path, context_path, metadata_path


def write_popularity_audit(
    data,
    index_to_itemid: np.ndarray,
) -> tuple[pd.DataFrame, Path]:
    """Rank items only by training-basket frequency, never using test counts."""
    output_path = MODEL_DIR / "popularity_full_catalogue_test_per_example.csv"
    metrics_path = MODEL_DIR / "popularity_full_catalogue_test_metrics.csv"
    if output_path.exists():
        output_path.unlink()

    frequencies = np.zeros(data.num_train_items + 1, dtype=np.int64)
    train_baskets = data.model_baskets.loc[data.model_baskets["split"].eq("train")]
    for item_json in train_baskets["item_indices"]:
        frequencies[np.asarray(json.loads(item_json), dtype=np.int64)] += 1

    if frequencies.sum() != int(train_baskets["known_item_count"].sum()):
        raise AssertionError("popularity must count training basket-item memberships only")

    item_indices = np.arange(1, data.num_train_items + 1)
    ordered_items = item_indices[np.lexsort((item_indices, -frequencies[1:]))]
    top_indices = ordered_items[:MAX_K]
    top_itemids = index_to_itemid[top_indices]
    ranks = np.empty(data.num_train_items + 1, dtype=np.int64)
    ranks[ordered_items] = np.arange(1, data.num_train_items + 1)

    totals = metric_state()
    start = time.perf_counter()
    target_lists = [json.loads(value) for value in data.test_examples["target_item_indices"]]
    all_rows = []

    for row_number, (_, row) in enumerate(data.test_examples.iterrows()):
        actual_indices = np.asarray(target_lists[row_number], dtype=np.int64)
        actual_itemids = [int(index_to_itemid[value]) for value in actual_indices]
        relevance = np.isin(top_indices, actual_indices)
        target_count = len(actual_indices)
        for k in K_VALUES:
            rel = relevance[:k]
            totals[k]["hit_count"] += int(rel.any())
            totals[k]["recall_sum"] += rel.sum() / target_count
            discount = 1.0 / np.log2(np.arange(2, k + 2))
            dcg = (rel * discount).sum()
            idcg = discount[: min(target_count, k)].sum()
            totals[k]["ndcg_sum"] += dcg / idcg

        best_actual_index = actual_indices[np.argmin(ranks[actual_indices])]
        retrieved = [int(itemid) for itemid, hit in zip(top_itemids, relevance) if hit]
        all_rows.append(
            {
                "context_row_index": row_number,
                "visitorid": int(row["visitorid"]),
                "target_timestamp": int(row["target_timestamp"]),
                "target_block_id": row["target_block_id"],
                "history_length": int(row["history_length"]),
                "actual_next_itemids": json.dumps(actual_itemids),
                "actual_item_count": target_count,
                "predicted_top_1_itemid": int(top_itemids[0]),
                "predicted_top_1_logit_score": int(frequencies[top_indices[0]]),
                "top_1_result": "TP" if relevance[0] else "MISS",
                "top_50_result": "TP_IN_TOP_50" if relevance.any() else "FAILED_TOP_50",
                "hit_at_10": bool(relevance[:10].any()),
                "hit_at_20": bool(relevance[:20].any()),
                "hit_at_50": bool(relevance.any()),
                "best_actual_itemid": int(index_to_itemid[best_actual_index]),
                "best_actual_logit_score": int(frequencies[best_actual_index]),
                "best_actual_full_catalogue_rank": int(ranks[best_actual_index]),
                "retrieved_actual_itemids_in_top_50": json.dumps(retrieved),
                "retrieved_actual_item_count_in_top_50": len(retrieved),
            }
        )

    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    elapsed = time.perf_counter() - start
    metrics = metrics_frame(totals, len(all_rows), "popularity", None, elapsed)
    metrics["catalogue_items"] = data.num_train_items
    metrics.to_csv(metrics_path, index=False)
    return metrics, output_path


def refresh_experiment_log(comparison: pd.DataFrame) -> None:
    """Record the completed fixed-test comparison without duplicating entries."""
    log_path = PROJECT_ROOT / "EXPERIMENT_LOG.md"
    if not log_path.exists():
        return

    def metric(model_name: str, k: int, column: str) -> float:
        row = comparison.loc[
            comparison["model"].eq(model_name) & comparison["k"].eq(k),
            column,
        ]
        return float(row.iloc[0])

    summary = (
        "<!-- TISASREC_FIXED_TEST_RESULT_START -->\n"
        "## TiSASRec fixed shared-test result — generated by Notebook 12\n\n"
        f"- Popularity: Hit@10 = {metric('popularity', 10, 'event_hit_rate'):.4f}, "
        f"NDCG@10 = {metric('popularity', 10, 'ndcg'):.4f}.\n"
        f"- SASRec: Hit@10 = {metric('sasrec', 10, 'event_hit_rate'):.4f}, "
        f"NDCG@10 = {metric('sasrec', 10, 'ndcg'):.4f}.\n"
        f"- TiSASRec: Hit@10 = {metric('tisasrec', 10, 'event_hit_rate'):.4f}, "
        f"NDCG@10 = {metric('tisasrec', 10, 'ndcg'):.4f}.\n"
        "- This is a fixed shared-test comparison. TiSASRec was selected on validation; "
        "no result here may be used for further model tuning.\n"
        "<!-- TISASREC_FIXED_TEST_RESULT_END -->\n"
    )
    text = log_path.read_text(encoding="utf-8")
    start_marker = "<!-- TISASREC_FIXED_TEST_RESULT_START -->"
    end_marker = "<!-- TISASREC_FIXED_TEST_RESULT_END -->"
    if start_marker in text and end_marker in text:
        before, remainder = text.split(start_marker, 1)
        _, after = remainder.split(end_marker, 1)
        text = before + summary + after.lstrip("\n")
    else:
        text = text.rstrip() + "\n\n" + summary
    log_path.write_text(text, encoding="utf-8")


def run_evaluation() -> dict[str, Path]:
    """Run TiSASRec, popularity, and SASRec full-catalogue comparison."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this configured evaluation.")
    device = torch.device("cuda")
    checkpoint_path = MODEL_DIR / "best_model_by_validation_ndcg10.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Train TiSASRec before requesting its final evaluation.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    data = load_basket_sequence_data(PROJECT_ROOT, int(config["max_history_baskets"]))
    if len(data.test_examples) != 173_786:
        raise AssertionError("unexpected modelable test example count")

    model = BasketTiSASRec(
        basket_encoder=BasketEncoder(data.num_train_items, int(config["embedding_dim"])),
        max_history_baskets=int(config["max_history_baskets"]),
        embedding_dim=int(config["embedding_dim"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        feed_forward_dim=int(config["feed_forward_dim"]),
        num_time_buckets=int(config["num_time_buckets"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = TimeAwareNextBasketDataset(
        data.test_examples,
        data.basket_item_matrix,
        data.basket_timestamps_ms,
        int(config["max_history_baskets"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    index_to_itemid = load_item_lookup(data.num_train_items)

    print("Running exact TiSASRec full-catalogue evaluation...")
    tisasrec_metrics, tisasrec_audit, context_path, metadata_path = write_tisasrec_audit(
        model,
        test_loader,
        data.test_examples,
        index_to_itemid,
        int(checkpoint["epoch"]),
        device,
    )

    print("Running training-only popularity baseline...")
    popularity_metrics, popularity_audit = write_popularity_audit(data, index_to_itemid)

    sasrec_metrics_path = SASREC_MODEL_DIR / "full_catalogue_test_metrics.csv"
    if not sasrec_metrics_path.exists():
        raise FileNotFoundError("Existing SASRec full-catalogue metrics are required for comparison.")
    sasrec_metrics = pd.read_csv(sasrec_metrics_path)
    sasrec_metrics["model"] = "sasrec"
    sasrec_metrics["elapsed_seconds"] = np.nan
    sasrec_metrics["checkpoint_epoch"] = 15
    sasrec_metrics["catalogue_items"] = data.num_train_items
    sasrec_metrics["evaluation_protocol"] = "full catalogue: every training-known item scored for every test example"

    columns = [
        "model", "evaluation_protocol", "checkpoint_epoch", "test_examples",
        "catalogue_items", "k", "event_hit_rate", "multi_positive_recall",
        "ndcg", "elapsed_seconds",
    ]
    comparison = pd.concat(
        [popularity_metrics[columns], sasrec_metrics[columns], tisasrec_metrics[columns]],
        ignore_index=True,
    ).sort_values(["k", "model"]).reset_index(drop=True)
    comparison_path = MODEL_DIR / "full_catalogue_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    refresh_experiment_log(comparison)

    print("✅ TiSASRec full-catalogue evaluation complete.")
    print("TiSASRec audit:", tisasrec_audit)
    print("Popularity audit:", popularity_audit)
    print("Comparison:", comparison_path)
    print(comparison.to_string(index=False))

    return {
        "tisasrec_audit": tisasrec_audit,
        "tisasrec_context_vectors": context_path,
        "tisasrec_context_metadata": metadata_path,
        "popularity_audit": popularity_audit,
        "comparison": comparison_path,
    }


if __name__ == "__main__":
    run_evaluation()
