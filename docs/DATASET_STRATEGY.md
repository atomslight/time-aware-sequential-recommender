# Dataset Strategy and Data Contract

## Primary benchmark — Retailrocket

Retailrocket is the active dataset. Notebook 07 prepares unordered time blocks
keyed by `(visitorid, timestamp)`. The planned prediction context contains only
strictly earlier blocks, and the target is the set of distinct items in the next
block. Labels, model representation, and evaluation protocol are not yet built.

The current eligible-event policy includes `view`, `addtocart`, and
`transaction`. All three are retained during preparation; this does not imply
equal business value. Event rows and distinct block-item memberships are saved
separately, so multiple actions on one item remain traceable.

Preparation removes only later exact copies and retains raw source-row numbers.
Unix millisecond timestamps remain unchanged. Block IDs are stable text keys
derived from visitor and timestamp; item sorting is storage/display order only.
A transaction basket requires multiple distinct items, all-transaction events,
and one shared nonblank transaction ID. Other simultaneous events are time blocks,
not necessarily purchases.

The earlier tie-exclusion rule and whole-timeline iterative 5-core output are
historical exploratory versions, superseded for main dataset preparation. Tie
exclusion lost 3,728 transaction events (16.6% of purchases). Notebook 07 retains
all 22,457 transaction events after exact-deduplication and applies no minimum-five
filter. Its outputs are prepared data, not yet training-ready.

Next, define global chronological train/validation/test boundaries, then determine
eligibility and the item catalog from training-period evidence only. Final thresholds
and whether eligibility counts events, distinct items, or blocks remain protocol
decisions. Report resulting coverage losses. Basket representation and multi-positive
ranking evaluation must be specified for both models; compatibility with unmodified
SASRec/TiSASRec is not assumed.

## Secondary benchmark — H&M

H&M is a deferred robustness study. It begins only after the Retailrocket
protocol and core comparison are frozen. Its transaction date has day-level
granularity. Therefore, purchases on one date form an unordered basket; history
may contain only purchases from earlier dates, and the next active date is the
multi-positive target basket. No within-day item order may be fabricated.

H&M results answer a related purchase-basket ranking question and must be
reported separately from Retailrocket next-event metrics. The project downloads
only `transactions_train.csv`, `articles.csv`, and `customers.csv`; images are
out of scope.

## Reproducibility record

Every future run must record: dataset/version, raw-file checksum, event or
basket policy, eligibility threshold, time boundaries, candidate protocol,
seed, configuration, metrics, and runtime/resources in `EXPERIMENT_LOG.md`.

