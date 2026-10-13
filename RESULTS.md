# Results

Measured from the supplied merged CSV. Historical guide figures are recorded separately in presentation/README.md.

## Dataset

| Metric | Value |
| --- | ---: |
| source_rows | 670,808 |
| source_columns | 14 |
| known_death_outcomes | 377,392 |
| death_reports | 27,904 |
| non_death_reports | 349,488 |
| unknown_death_outcomes | 293,416 |
| exact_duplicate_source_rows | 80,659 |
| hospitalization_flagged | 81,362 |
| disabling_flagged | 6,134 |
| invalid_or_missing_weight | 602,178 |
| invalid_or_missing_date | 277,909 |

## Evaluation

| Run | Model | Accuracy | Death precision | Death recall | Death F1 | Average precision | Fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Majority Dummy | 0.9261 | 0.0000 | 0.0000 | 0.0000 | 0.0739 | 0.01 |
| Baseline | Logistic Regression | 0.9628 | 0.9037 | 0.5567 | 0.6890 | 0.7714 | 3.71 |
| Baseline | Random Forest | 0.9561 | 0.9899 | 0.4100 | 0.5798 | 0.7700 | 0.90 |
| Baseline | Linear SVM | 0.9618 | 0.8867 | 0.5536 | 0.6816 | 0.7613 | 0.31 |
| Baseline | KNN | 0.9502 | 0.8091 | 0.4277 | 0.5596 | 0.5353 | 0.01 |
| SMOTENC | Majority Dummy | 0.9260 | 0.0000 | 0.0000 | 0.0000 | 0.0740 | 0.00 |
| SMOTENC | Logistic Regression | 0.8867 | 0.3621 | 0.6982 | 0.4769 | 0.6622 | 0.96 |
| SMOTENC | Random Forest | 0.9183 | 0.4605 | 0.6036 | 0.5224 | 0.6233 | 0.83 |
| SMOTENC | Linear SVM | 0.8810 | 0.3476 | 0.6937 | 0.4632 | 0.6482 | 0.06 |
| SMOTENC | KNN | 0.7280 | 0.1793 | 0.7477 | 0.2892 | 0.2525 | 0.00 |

## Measured observations

Logistic Regression has the highest baseline death F1 in this run (0.6890), with accuracy 96.28%, death precision 90.37%, and death recall 55.67%. This ranking describes this particular split and is not a clinical model recommendation.

Logistic Regression: death recall changes by +14.15 percentage points between the two runs. Baseline F1 is 0.6890; SMOTENC F1 is 0.4769.

Random Forest: death recall changes by +19.36 percentage points between the two runs. Baseline F1 is 0.5798; SMOTENC F1 is 0.5224.

Linear SVM: death recall changes by +14.01 percentage points between the two runs. Baseline F1 is 0.6816; SMOTENC F1 is 0.4632.

KNN: death recall changes by +32.01 percentage points between the two runs. Baseline F1 is 0.5596; SMOTENC F1 is 0.2892.


## Interpretation and limits

Baseline uses up to 65,000 known-target rows; SMOTENC uses up to 15,000 before resampling. Each run has its own untouched stratified 20% test split. Differences combine sample-size effects with balancing effects, so this is not a controlled estimate of the effect of SMOTENC. For a controlled comparison, rerun both with the same sample size and seed.

Rare categories, imputation, scaling and encoding are fitted on training rows only. SMOTENC sees training rows only. Numeric fields are standardized before its neighbor search. Hospitalization, disability and death codes are excluded from predictors.

These are retrospective classifications of reports with known outcome codes, not prospective predictions of patient mortality or evidence that a drug caused death. Reactions may already describe an outcome. Spontaneous report counts are not drug incidence rates. The merged file has no report identifier: repeated reports, follow-up versions and shared patients cannot reliably be grouped, and overlapping records may inflate random-split performance. Unknown outcomes are excluded and may introduce selection bias. No clinical validation, confidence intervals or external test cohort is provided.

Exact source duplicates are counted but retained to preserve the documented source experiment. Review deduplication and report-level identifiers before making substantive conclusions. All convergence warnings are saved in each run JSON; warning flags also appear in metrics CSVs. Models are trusted local joblib artifacts; do not load untrusted joblib files.
