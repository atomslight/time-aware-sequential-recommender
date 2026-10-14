# Experiment Log

Record every run that produces a metric or influences a modeling decision. Do not overwrite failed or negative results.

| Experiment ID | Date | Hypothesis | Dataset / split | Configuration | Seed | Metrics | Runtime / resources | Observation / outcome |
|---|---|---|---|---|---:|---|---|---|
| — | — | — | — | — | — | — | — | No experiments have been run. |

## Model runs

| Experiment ID | Date | Hypothesis | Dataset / split | Configuration | Seed | Metrics | Runtime / resources | Observation / outcome |
|---|---|---|---|---|---:|---|---|---|
| E-001 | 2026-10-14 | Sequential basket context improves next-item ranking over simple global popularity. | Fixed basket-aware Retailrocket test set; 173,786 modelable examples; 200,978-item training catalogue. | Basket-aware SASRec, 64 dimensions, 2 layers, 2 heads, 20-basket history, multi-positive sampled loss. | 42 | Full-catalogue Hit@1 0.3163; Hit@10 0.4482; NDCG@10 0.3811; Hit@50 0.5149. | RTX 4070 Laptop GPU; exact full-catalogue audit completed. | This replaces the earlier sampled-candidate interpretation for final reporting. The test split is now a fixed shared comparison, not blind. |
| E-002 | 2026-10-14 | Explicit relative time intervals improve or match SASRec under the same data and evaluation rules. | Same fixed basket-aware Retailrocket split. | Basket-aware TiSASRec; 32 log-bucketed relative-time intervals; selection by validation full-catalogue NDCG@10. | 42 | Pending Notebook 11 training and Notebook 12 evaluation. | RTX 4070 Laptop GPU. | Do not tune after the fixed shared-test evaluation. |

## Data inspection

| Inspection ID | Date | Dataset | Policy | Key result | Outcome |
|---|---|---|---|---|---|
| I-001 | 2026-10-12 | Retailrocket `events.csv` | All event types; minimum 5 events per visitor | 81,620 retained users, 948,537 retained events, 103,873 retained items; varied within-user gaps | Primary benchmark is viable with explicit active-user scope. No model or evaluation metric has been run. |
| I-002 | 2026-10-13 | Retailrocket `events.csv` | Keep the first row of every fully identical event; then measure, but do not remove, repeated visitor--item interactions | 460 exact-copy rows removed into a separate cleaned file. 332,884 visitor--item pairs repeat at least twice after this step. | Exact copies and real repeat behaviour are different. No repeat-interaction policy has been chosen yet. |
| I-003 | 2026-10-13 | Retailrocket `events.csv` and the separate exact-deduplicated copy | Assumption check before sequence construction | Required sequence fields are complete; event names and transaction-ID blanks behave as expected; cleaned-row count reconciles exactly. 5,686 rows participate in a visitor-and-timestamp tie. | Continue only after a deterministic timestamp-tie policy is selected and documented. |
| I-004 | 2026-10-13 | Exact-deduplicated Retailrocket events | Exclude every member of a visitor-and-timestamp tie from the modeling copy | 4,769 tied events in 2,143 groups excluded; 2,750,872 events remain with zero timestamp ties and zero exact duplicates. | The next-item order is no longer invented within same-timestamp groups. |
| I-005 | 2026-10-13 | Tie-free Retailrocket modeling copy | Compare five candidate eligible-event policies with a simple per-user >=5 event check | All events: 81,484 eligible visitors and 943,099 events. Cart+transaction: 2,221 visitors. Transaction-only: 238 visitors. All-events gaps range from a 0.03-hour median to a 903.72-hour 99th percentile. | Lock all three event types for the core next-eligible-item benchmark. |
| I-006 | 2026-10-13 | Tie-free all-event Retailrocket modeling copy | Iteratively require >=5 interactions per visitor and >=5 per item | Seven loops; final stable output: 775,803 events, 65,815 visitors, 35,996 items. Minimum final count for both visitors and items: 5. | Use this separate 5-core output as the input to chronological sequence construction. |

## Notebook 07 preparation status — 2026-10-13

- Authored `07_build_basket_dataset.ipynb`; its full production run is left to the user in JupyterLab.
- Syntax, notebook JSON/structure, configuration JSON, grouping examples, and shuffled-input invariance passed. The two existing project tests also passed.
- Expected source-version checks are 2,755,641 deduplicated events, 22,457 transaction events, 2,753,015 blocks, and 2,143 tied blocks containing 4,769 events. These are prior audit expectations, not results from a Notebook 07 production run.
- The notebook will record input checksums, package versions, output sizes/row counts, settings, CPU use, and elapsed time in each run's manifest. Only a successfully completed run receives status `prepared data; not yet training-ready`.
- I-004 and I-006 remain historical explorations. Their tie-excluded/whole-timeline outputs are no longer the main preparation path.

## Future run checklist

- Commit or record the exact code version.
- Record the dataset version, preprocessing configuration, split boundaries, candidate protocol, and random seed.
- Report ranking metrics at the same K values for every comparable model.
- Mark results as non-comparable if candidate sets or evaluation populations differ.

<!-- TISASREC_FIXED_TEST_RESULT_START -->
## TiSASRec fixed shared-test result — generated by Notebook 12

- Popularity: Hit@10 = 0.0046, NDCG@10 = 0.0021.
- SASRec: Hit@10 = 0.4482, NDCG@10 = 0.3811.
- TiSASRec: Hit@10 = 0.4584, NDCG@10 = 0.3898.
- This is a fixed shared-test comparison. TiSASRec was selected on validation; no result here may be used for further model tuning.
<!-- TISASREC_FIXED_TEST_RESULT_END -->
