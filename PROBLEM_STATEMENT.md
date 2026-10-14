# TemporalRec — Problem Statement (Draft)

## Objective

Evaluate whether adding elapsed-time information to an ordered customer interaction history improves **next-item ranking** over order-only sequential modeling and simple popularity baselines.

## Prediction unit

At a prediction cutoff \(t\) for a user \(u\):

- **Context:** the visitor's time blocks strictly before \(t\), ordered by timestamp; items inside each block are unordered.
- **Target:** the distinct items in the next eligible timestamp block (potentially multiple positives).
- **Output:** a ranked list of candidate items, not a probability for a single independently sampled row.
- **Evaluation:** ranking quality for the target item set; precise multi-positive metric definitions remain a protocol decision.

## Primary task definition — basket direction selected; model protocol pending

| Concept | Retailrocket working interpretation |
|---|---|
| User/customer | `visitorid` |
| Interaction | timestamped product event, potentially a view, add-to-cart, or transaction |
| Item | `itemid` |
| Available information | eligible events that occurred strictly before the prediction cutoff only |
| Input \(X\) | strictly earlier time blocks, each containing an unordered set of items; model encoding not yet selected |
| Target \(y\) | the set of distinct item IDs in the next eligible timestamp block; labels not yet constructed |

The eligible-event policy includes `view`, `addtocart`, and `transaction`. Exact
duplicates are reduced to one copy. Simultaneous events are preserved in time
blocks. The former single-item target and tie-exclusion preparation are historical
versions. The basket direction requires explicit adaptations before comparing
SASRec and TiSASRec; unmodified model compatibility is not claimed.

## Research hypotheses

- **H1:** SASRec outperforms global/recent popularity on temporally held-out next-item ranking for eligible users.
- **H2:** TiSASRec improves ranking metrics over SASRec when users exhibit varied and informative time gaps.
- **H3:** Any time-aware benefit is heterogeneous, varying by history length, interaction density, and elapsed time since the prior event.
- **H0:** Explicit time intervals do not yield a reliable improvement under the same data, split, candidates, tuning budget, and seeds.

## Scope and limitations

- This is an offline retail-event benchmark, not evidence of corporate-card cross-sell lift.
- Ranking accuracy does not establish causality, campaign response, product eligibility, or fairness.
- We will not begin training until data suitability and a leakage-safe temporal evaluation design are validated.

