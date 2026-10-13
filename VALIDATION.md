# Validation

The complete pipeline was executed locally against the supplied 670,808-row, 14-column CSV with seed 42, a 65,000-row baseline sample and a 15,000-row SMOTENC sample. All four models completed in both experiments. The matching original presentation was preserved.

Six automated tests pass, covering invalid weights and dates, unknown outcomes, source-schema validation, exclusion of outcome fields from predictors, training-only preprocessing, code/date preservation in synthetic generation, refusal to append synthetic experiments, train/test separation, resampling and saved-model inference.

`tools/verify_outputs.py` checks the 10 metric rows including dummy references, confusion-matrix totals, disjoint source-row splits, balanced SMOTENC training, all eight serialized model bundles against held-out predictions, notebook Python cell syntax and all 30 PNG files. A separate command-line inference run successfully classified the 200-row sample.

Plot images were reviewed for readable axes, labels and confusion matrices. The valid source dates cover one transmission month; the monthly plot displays that single observed month rather than implying a longer trend.

The Colab notebook has 20 cells and embeds the shared module. Its Python cells were compiled locally; it was not executed in a live Colab runtime. The optional synthetic generator was validated on a small file; the optional 1 GB experiment was left disabled as specified by the guide.

The archive is integrity-checked and excludes the full raw/processed/synthetic datasets and local Python environment. `.gitignore` was verified for full datasets, the environment and model binaries. No GitHub repository was created or published.

## Repeat checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools/verify_outputs.py
```

Run the output verifier after a full analysis run using the default real dataset. It assumes all expected results and model bundles are present.
