# Data

`raw/2025_merged.csv` is the supplied full dataset copied locally from Downloads. `processed/2025_merged_clean.csv` is generated. Both are ignored by Git. `sample_patient_drug_events.csv` contains 200 actual source rows sampled with seed 42, not synthetic records. It demonstrates the schema and should not be used to claim meaningful model performance.

The code expects the guide's 14 source fields. Death coding is specific to this merged course dataset: 1 is death, 2 is non-death, and other codes are unknown. Missing death values must not be treated as non-death. The file does not contain Age or report identifiers. No redistribution license or independent OpenFDA record lineage was supplied.

Place a replacement full CSV in `raw/`, or pass `--data` to the CLI. Never publish full raw, processed or synthetic data accidentally.
