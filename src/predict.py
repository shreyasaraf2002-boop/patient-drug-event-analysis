"""Classify reports using a trusted saved local model."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
from src.project import clean_data, read_raw, FEATURES, NUM


def predict(bundle, raw):
    clean = clean_data(raw)
    prepared = bundle['prepare'].transform(clean[FEATURES])
    if bundle['numeric_scale'] is not None:
        prepared[NUM] = bundle['numeric_scale'].transform(prepared[NUM])
    encoded = bundle['encoder'].transform(prepared)
    output = pd.DataFrame({'source_row': raw.index, 'predicted_death_report': bundle['model'].predict(encoded)})
    if hasattr(bundle['model'], 'predict_proba'):
        output['model_score'] = bundle['model'].predict_proba(encoded)[:, 1]
    else:
        output['model_score'] = bundle['model'].decision_function(encoded)
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('outputs/results/new_predictions.csv'))
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    predict(joblib.load(a.model), read_raw(a.data)).to_csv(a.output, index=False)
    print(a.output)
