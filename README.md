# TemporalRec

Research project for testing whether time-interval modeling improves next-item ranking.

## Current phase

Retailrocket is the primary benchmark. H&M is intentionally deferred until the
Retailrocket protocol, models, and evaluation are complete.

Notebook 07 is the active preparation path: retain exact-deduplicated events in
unordered visitor/timestamp blocks. Earlier strict tie-exclusion and whole-timeline
5-core outputs are historical exploratory versions, not the final training dataset.

## Dataset inspection

The initial inspection is read-only and does **not** train a model or create
labels. From the project root, run:

```powershell
$env:PYTHONPATH = "src"
python scripts/inspect_retailrocket.py --events dataset/events.csv
python -m unittest discover -s tests -v
```

Use `--output reports/retailrocket_inspection.json` to save a machine-readable
report. Generated reports and future raw H&M files are ignored by Git.

See `docs/DATASET_STRATEGY.md` for the locked study scope and data contracts.

## Learning notebooks

Run [01_data_quality_cleaning.ipynb](notebooks/01_data_quality_cleaning.ipynb)
in JupyterLab to reproduce the data-quality checks and create separate,
versioned CSV audits. It does not modify the raw Retailrocket file.

Then run [02_event_policy_eda.ipynb](notebooks/02_event_policy_eda.ipynb) to
compare the possible `view`, `addtocart`, and `transaction` sequence policies
before making the final choice.

After the event policy is locked, run
[03_iterative_5core_filter.ipynb](notebooks/03_iterative_5core_filter.ipynb)
to apply and audit the iterative user-and-item minimum-five interaction rule.

Run [04_visual_eda.ipynb](notebooks/04_visual_eda.ipynb) after the 5-core
filter to generate recommendation-specific charts before chronological splitting.

Run [05_cleaning_impact_and_repeats.ipynb](notebooks/05_cleaning_impact_and_repeats.ipynb)
to compare cleaning losses by event type and consecutive same-item behavior.
Review these results and the full-timeline eligibility limitation before split design.

Run [06_timestamp_basket_tradeoff.ipynb](notebooks/06_timestamp_basket_tradeoff.ipynb)
to audit same-timestamp transaction baskets and compare exclusion with a
basket-aware policy before changing the sequence design.

Run [07_build_basket_dataset.ipynb](notebooks/07_build_basket_dataset.ipynb) to
write events with block IDs, distinct block-item memberships, block summaries,
and Excel audits into new timestamped folders. Notebook 01's exact-deduplicated
CSV is required for verification. Run cells top to bottom; completion is recorded
in the run manifest. Large dataset CSVs exceed Excel's worksheet limit; use the
small audit files for manual checks. Temporal splits, training-only eligibility,
and model representation follow separately. No production run of Notebook 07 is
implied by adding or testing the notebook.

