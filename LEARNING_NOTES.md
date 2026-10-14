# Learning Notes

## 2026-10-13 — Basket preparation replaces the strict exploratory path

- The earlier tie-exclusion rule removed 3,728 purchase events, or 16.6% of purchases, even though the overall event loss was only 0.17%.
- Keep all retained event rows in one table, and list each distinct item once per visitor/timestamp block in another. A view and cart action can remain two events but contribute one possible target item within that block.
- A stable block ID comes from visitor and timestamp. Sorting by item ID only makes saved files reproducible; it never tells us which item came first.
- Notebook 07 prepares data and audit files. It does not train, split, or apply the minimum-five rule. The earlier whole-timeline 5-core result is exploratory because future activity influenced eligibility.
- A future example can have several correct target items. History must use strictly earlier blocks. Both model input representation and multi-positive evaluation need explicit design before training.

Capture concise concepts, surprises, issues, and resolutions in language you can reuse in an interview.

## 2026-10-12 — Project framing

- **Sequential recommendation** predicts a future item from an ordered interaction history; it is not ordinary row-wise classification.
- **Time-aware sequential recommendation** asks whether elapsed time between events adds predictive signal beyond their order.
- **Key validity risk:** future interactions must never influence the history, candidate universe, statistics, preprocessing fit, or model selection for an earlier prediction.
- **Open question:** whether Retailrocket has enough usable chronological sequences and varied time gaps after leakage-safe filtering.

## 2026-10-12 — Dataset suitability check

- Retailrocket's timestamps are Unix milliseconds and cover about 138 days.
- Within-user gaps vary substantially (median about 2.3 minutes; 90th percentile about 73 hours), so elapsed-time information is plausibly useful.
- Visitor histories are sparse (median = 1 event). A sequential-recommendation benchmark will necessarily represent an eligible, more-active subset; this selection must be measured and disclosed.

## 2026-10-12 — Initial inspection result

- Requiring five eligible events retains 81,620 visitors and 948,537 events: a workable core benchmark, but only 5.80% of all visitors.
- Exact timestamp ties occur for 5,686 rows within a visitor. The current implementation preserves them and counts them; we will decide a non-arbitrary tie policy before labels or temporal splits are created.
- There are 460 fully duplicated raw rows. A duplicate-event policy remains a preprocessing decision, not an implicit cleanup step.

## 2026-10-13 — Locked primary task

- A raw `itemid` becomes the label `next_item_id` only when it is the next chronological event for the same visitor.
- The same `itemid` may appear in an earlier history for one prediction and be the future target for an earlier prediction.
- Every feature in a prediction example must come from events strictly earlier than its target event.

## 2026-10-13 — Exact-duplicate policy

- An exact duplicate means the five raw event fields match: timestamp, visitor ID, event type, item ID, and transaction ID.
- We keep the earliest source row in each duplicate group and remove only later exact copies in a new cleaned file.
- A repeated visit to the same item at a different time is not an exact duplicate and remains in the data.

## 2026-10-13 — Repeated interactions

- After exact copies are removed, a visitor can still interact with the same item several times. This can be normal behaviour, not bad data.
- We found 332,884 visitor--item pairs with at least two interactions. We measured them first and deliberately did not delete them.

## 2026-10-13 — Assumption check

- A data check can pass while a design decision is still needed. Here, required fields are complete and the cleaned file reconciles exactly, but 5,686 rows share a visitor and timestamp.
- Before making a "next item" label, we must define how tied events are ordered. Otherwise, we could invent an order that the source data did not record.

## 2026-10-13 — Timestamp-tie policy

- We chose not to use file row order as a fake clock. Instead, every event in a same-visitor, same-timestamp group is excluded from the modeling copy.
- This removes only 4,769 of 2,755,641 exact-deduplicated events (about 0.17%) and leaves no tied timestamps for the sequence model.

## 2026-10-13 — Event-policy choice

- `view` is 96.81% of the cleaned events, but it is still useful behavior for a next-interaction task. Removing it would greatly reduce the number of usable sequences.
- Transaction-only data leaves just 238 visitors with at least five purchases, so it is too small for the primary sequential-model comparison.
- We keep all three event types because it matches the stated target: the next eligible interaction, not only the next purchase.

## 2026-10-13 — Iterative 5-core filter

- A one-time user filter is not enough. When rare items are removed, some users lose events and may fall below the minimum; this is why the filter repeats.
- The loop stopped only when it removed zero rows. The stable benchmark has 65,815 visitors, 35,996 items, and 775,803 events, with every retained visitor and item appearing at least five times.

## 2026-10-14 — Sampled candidates versus full catalogue

- A sampled-candidate metric asks whether the real item beats a limited set of randomly drawn wrong items. It is useful during fast training, but can look much higher than a production-like score.
- A full-catalogue metric ranks the real next item against every item the model knows. For the current benchmark, that means 200,978 training-known items per test example.
- The SASRec full-catalogue result is the reportable number: Hit@1 = 31.63%, Hit@10 = 44.82%, NDCG@10 = 38.11%, and Hit@50 = 51.49%.
- A long tail is normal in recommendation: the median best true-item rank was 35, while difficult interactions can rank far lower. A single failed example does not describe the whole model.

## 2026-10-14 — Why TiSASRec is a fair next comparison

- SASRec knows the order of earlier baskets. TiSASRec additionally receives the elapsed time between them.
- A log time bucket lets the model distinguish seconds, minutes, hours, and days without needing one embedding for every possible millisecond gap.
- TiSASRec will use the same baskets, chronology, item catalogue, targets, GPU budget, and full-catalogue evaluation as SASRec. This makes a later difference in score attributable to the time-aware attention design rather than a changed dataset.

