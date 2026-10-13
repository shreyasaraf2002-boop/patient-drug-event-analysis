import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.project import (CAT, FEATURES, NUM, SOURCE, TARGET, ReportPreparer,
                         clean_data, folders, read_raw, subset, synthetic, train)
from src.predict import predict


def fixture(n=160):
    df = pd.DataFrame({c: ['example'] * n for c in SOURCE})
    df['Gender'] = [1, 2] * (n // 2)
    df['Weight'] = [55, 70] * (n // 2)
    df['Seriousness_Death'] = [1 if i % 8 == 0 else 2 for i in range(n)]
    df['Seriousness_Disabling'] = 2
    df['Seriousness_Hospitalization'] = 2
    df['transmissiondate'] = 20250716
    return df


class PipelineTests(unittest.TestCase):
    def test_cleaning_keeps_unknown_target_and_handles_bad_values(self):
        raw = fixture()
        raw['Weight'] = raw['Weight'].astype(object)
        raw['transmissiondate'] = raw['transmissiondate'].astype(float)
        raw.loc[:3, 'Weight'] = [0, 651, 'bad', 650]
        raw.loc[:2, 'Seriousness_Death'] = [1, 2, 9]
        raw.loc[0, 'transmissiondate'] = 20250230
        raw.loc[1, 'transmissiondate'] = 20250716.5
        raw.loc[0, 'Reaction_0'] = '  '
        clean = clean_data(raw)
        self.assertTrue(clean.loc[:2, 'Weight'].isna().all())
        self.assertEqual(clean.loc[3, 'Weight'], 650)
        self.assertEqual(clean.loc[0, TARGET], 1)
        self.assertEqual(clean.loc[1, TARGET], 0)
        self.assertTrue(pd.isna(clean.loc[2, TARGET]))
        self.assertTrue(clean.loc[:1, 'Transmission_Date'].isna().all())
        self.assertEqual(clean.loc[0, 'Reaction_0'], 'Unknown')
        self.assertFalse(any(c.startswith('Seriousness') for c in FEATURES))

    def test_preparer_does_not_learn_from_test(self):
        clean = clean_data(fixture())[FEATURES]
        training, testing = clean.iloc[:100].copy(), clean.iloc[100:].copy()
        training['Weight'] = 60
        testing['Weight'] = np.nan
        testing['Country'] = 'TEST_ONLY'
        p = ReportPreparer(25).fit(training)
        result = p.transform(testing)
        self.assertTrue(result['Weight'].eq(60).all())
        self.assertTrue(result['Country'].eq('__RARE__').all())
        self.assertNotIn('TEST_ONLY', p.frequent_['Country'])

    def test_schema_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.csv'
            fixture().drop(columns='Gender').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Gender'):
                read_raw(path)

    def test_unknown_targets_excluded(self):
        clean = clean_data(fixture())
        clean.loc[:9, TARGET] = np.nan
        self.assertFalse(subset(clean, 80, 42)[TARGET].isna().any())

    def test_synthetic_preserves_codes_dates_and_refuses_append(self):
        raw = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'synthetic.csv'
            synthetic(raw, path, 1000, chunk_rows=20)
            generated = pd.read_csv(path)
            self.assertTrue(generated['transmissiondate'].eq(20250716).all())
            self.assertTrue(generated['Seriousness_Death'].isin([1, 2]).all())
            self.assertTrue(generated['Weight'].between(.1, 650).all())
            self.assertGreaterEqual(path.stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                synthetic(raw, path, 1000)

    def test_models_resampling_and_saved_inference(self):
        raw = fixture()
        clean = clean_data(raw)
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            folders(base)
            for run in ['Baseline', 'SMOTENC']:
                results = train(clean, run, 160, base=base, rare_threshold=2)
                self.assertEqual(len(results), 5)
                split = pd.read_csv(base / f'outputs/results/{run.lower()}_split_rows.csv')
                self.assertEqual(len(split), split.source_row.nunique())
                self.assertTrue(results[['tn', 'fp', 'fn', 'tp']].sum(axis=1).eq(32).all())
                import json
                info = json.loads((base / f'outputs/results/{run.lower()}_run.json').read_text())
                if run == 'SMOTENC':
                    self.assertEqual(len(set(info['train_after_resampling'].values())), 1)
                bundle = joblib.load(base / f'models/{run.lower()}_logistic_regression.joblib')
                output = predict(bundle, raw.iloc[:4])
                self.assertEqual(len(output), 4)
                self.assertTrue(output.predicted_death_report.isin([0, 1]).all())
                saved = pd.read_csv(base / f'outputs/results/{run.lower()}_logistic_regression_predictions.csv').head(4)
                restored = predict(bundle, raw.iloc[saved.source_row.to_numpy()])
                self.assertTrue(np.array_equal(restored.predicted_death_report, saved.predicted))
                self.assertTrue(np.allclose(restored.model_score, saved.score))


if __name__ == '__main__':
    unittest.main()
