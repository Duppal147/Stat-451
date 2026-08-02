# Which Measurements Diagnose a Breast Mass?

STAT 451 group project — Amoolya, Bonnie, Dhruvika, Kalynn, Edward.

Which of the 30 nuclear measurements in the Breast Cancer Wisconsin
(Diagnostic) data separate malignant masses from benign ones, and how few of
them do you actually need?

## Running it

```sh
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the three steps in order. Run it rather than the build
scripts on their own: they read whatever `results.json` is on disk, so building
without re-running the analysis is how a stale number reaches the report.

| File | What it does |
| --- | --- |
| `analysis.py` | Fits the models; writes `results.json` and `figures/*.png` |
| `build_report.py` | Writes `report.html`; prints the word count and flags >600 |
| `build_slides.py` | Writes `presentation.html` (arrow keys or click to advance) |

Every number quoted in the report and the slides is read from `results.json`,
so nothing has to be retyped when the analysis changes.

## Data

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer
Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5DW2B>

The first run downloads the data with `ucimlrepo` and caches it to `wdbc.csv`;
later runs read the cache, so they work offline. Delete `wdbc.csv` to re-fetch.

## Protocol

One stratified 80/20 split. Model choice, hyperparameter tuning, feature
ranking and the decision cutoff all come from 5-fold cross-validation *inside
the training set*. The test set is opened once, at the end, to score the two
models fixed before that pass: the full-feature SVM and the reduced model on
the six named features. Nothing is re-tuned afterwards.
