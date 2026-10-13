"""Audit generated outputs, notebook syntax and saved-model round trips."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.project import read_raw
from src.predict import predict

raw = read_raw(BASE / 'data/raw/2025_merged.csv')
checks = {}
comparison = pd.read_csv(BASE / 'outputs/results/model_comparison.csv')
assert len(comparison) == 10
for run in ['baseline', 'smotenc']:
    split = pd.read_csv(BASE / f'outputs/results/{run}_split_rows.csv')
    assert split.source_row.is_unique, 'Overlapping training/test source rows'
    metrics = pd.read_csv(BASE / f'outputs/results/{run}_metrics.csv')
    assert metrics[['tn', 'fp', 'fn', 'tp']].sum(axis=1).eq(metrics.test_rows).all()
    config = json.loads((BASE / f'outputs/results/{run}_run.json').read_text())
    if run == 'smotenc':
        assert len(set(config['train_after_resampling'].values())) == 1
    for path in sorted((BASE / 'models').glob(f'{run}_*.joblib')):
        bundle = joblib.load(path)
        assert type(bundle['prepare']).__module__ == 'src.project'
        saved = pd.read_csv(BASE / f'outputs/results/{path.stem}_predictions.csv').head(12)
        new = predict(bundle, raw.iloc[saved.source_row.to_numpy()])
        assert np.array_equal(new.predicted_death_report, saved.predicted)
        assert np.allclose(new.model_score, saved.score)
        checks[path.name] = 'Saved-model inference matches recorded held-out predictions'

notebook = json.loads((BASE / 'notebooks/Patient_Drug_Event_Analysis_Colab.ipynb').read_text())
assert notebook['nbformat'] == 4
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if not source.startswith('%pip'):
            compile(source, '<notebook cell>', 'exec')
checks['notebook'] = f'{len(notebook["cells"])} cells; Python cells compile; source embedded'

plots = sorted((BASE / 'outputs/plots').glob('*.png'))
assert len(plots) == 30, f'Expected 30 plots, found {len(plots)}'
sheet = Image.new('RGB', (1600, 6 * 290), 'white')
draw = ImageDraw.Draw(sheet)
for index, path in enumerate(plots):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        thumb = im.convert('RGB')
        thumb.thumbnail((310, 250))
        x, y = (index % 5) * 320, (index // 5) * 290
        sheet.paste(thumb, (x + (320 - thumb.width) // 2, y))
        draw.text((x + 4, y + 254), path.stem[:43], fill='black')
log = BASE / 'outputs/logs'
log.mkdir(exist_ok=True)
sheet.save(log / 'plot_contact_sheet.png')
checks['plots'] = f'{len(plots)} valid PNG figures'
checks['evaluation'] = '10 metric rows; disjoint source row splits; balanced SMOTENC training'
(log / 'verification.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
