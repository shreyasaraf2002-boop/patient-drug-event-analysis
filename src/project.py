"""Reproducible report-level analysis. Run python -m src.project --help."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import warnings
from pathlib import Path

import joblib
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'outputs' / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parents[1]
TARGET = 'Seriousness_Death_Binary'
NUM = ['Weight', 'Year', 'Month', 'Day']
CAT = ['Country', 'Gender_Label', 'Reaction_0', 'Reaction_1', 'Reaction_2',
       'Sender_Org', 'Chemical_Compound', 'Drug_Indication', 'Drug_Brand_Name']
FEATURES = CAT + NUM
SOURCE = ['Country', 'Gender', 'Weight', 'Reaction_0', 'Reaction_1', 'Reaction_2',
          'Sender_Org', 'Seriousness_Death', 'Seriousness_Disabling',
          'Seriousness_Hospitalization', 'transmissiondate', 'Chemical_Compound',
          'Drug_Indication', 'Drug/Brand name']
SERIOUS = ['Seriousness_Death', 'Seriousness_Disabling', 'Seriousness_Hospitalization']


def folders(base=BASE):
    for p in ['data/raw', 'data/processed', 'data/synthetic', 'outputs/plots',
              'outputs/results', 'models', 'presentation', 'notebooks']:
        (base / p).mkdir(parents=True, exist_ok=True)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def read_raw(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed', case=False)]
    missing = sorted(set(SOURCE) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required source columns: {missing}')
    if df.empty:
        raise ValueError('The input CSV contains no reports.')
    return df


def inspect(df, base=BASE):
    result = base / 'outputs/results'
    pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing_count': df.isna().sum(),
                  'missing_percent': df.isna().mean().mul(100),
                  'unique_values': df.nunique(dropna=True)}).to_csv(result / 'source_column_profile.csv')
    df['Seriousness_Death'].value_counts(dropna=False).rename_axis('source_code').to_csv(
        result / 'source_death_codes.csv', header=['reports'])
    print(f'Source shape: {df.shape}', flush=True)


def clean_data(df):
    clean = df.rename(columns={'Drug/Brand name': 'Drug_Brand_Name',
                               'transmissiondate': 'Transmission_Date_Raw'}).copy()
    for col in clean.select_dtypes(include=['object', 'string']).columns:
        clean[col] = clean[col].astype('string').str.strip().replace('', pd.NA)
    weight = pd.to_numeric(clean['Weight'], errors='coerce')
    clean['Weight'] = weight.where(weight.between(0, 650, inclusive='right'))
    clean['Gender'] = pd.to_numeric(clean['Gender'], errors='coerce')
    clean['Gender_Label'] = clean['Gender'].map({1: 'Male', 2: 'Female'}).fillna('Unknown')
    for col in SERIOUS:
        clean[col] = pd.to_numeric(clean[col], errors='coerce')
        clean[col + '_Binary'] = clean[col].map({1: 1, 2: 0})
    date = pd.to_numeric(clean['Transmission_Date_Raw'], errors='coerce')
    date = date.where(date.mod(1).eq(0)).astype('Int64').astype('string')
    clean['Transmission_Date'] = pd.to_datetime(date, format='%Y%m%d', errors='coerce')
    for col, accessor in [('Year', 'year'), ('Month', 'month'), ('Day', 'day')]:
        clean[col] = getattr(clean['Transmission_Date'].dt, accessor)
    for col in CAT:
        clean[col] = clean[col].fillna('Unknown').astype(str).str.strip()
    return clean


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def bar(series, title, filename, base=BASE, ylabel='Reports'):
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(series):
        ax.bar(range(len(series)), series.values, color='#2563a6')
        ax.set_xticks(range(len(series)), [str(v) for v in series.index], rotation=45, ha='right')
    else:
        ax.text(.5, .5, 'No eligible reports', ha='center', transform=ax.transAxes)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    save_figure(fig, base / 'outputs/plots' / filename)


def known_counts(df, col, n=15):
    return df.loc[df[col].ne('Unknown'), col].value_counts().head(n)


def eda(df, base=BASE):
    bar(known_counts(df, 'Gender_Label'), 'Known patient gender', 'gender_distribution.png', base)
    for col, stem in [('Country', 'countries'), ('Drug_Brand_Name', 'drugs'), ('Reaction_0', 'reactions')]:
        counts = known_counts(df, col)
        counts.to_csv(base / f'outputs/results/top_{stem}.csv', header=['reports'])
        bar(counts, f'Top reported {stem}', f'top_{stem}.png', base)
    monthly = df.dropna(subset=['Transmission_Date']).set_index('Transmission_Date').resample('ME').size()
    monthly.to_csv(base / 'outputs/results/monthly_reports.csv', header=['reports'])
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(monthly) == 1:
        ax.bar(monthly.index.strftime('%Y-%m'), monthly.values, color='#2563a6')
    elif len(monthly):
        ax.plot(monthly.index, monthly.values, marker='o', color='#2563a6')
    ax.set(title='Reports by transmission month', ylabel='Reports', xlabel='Month')
    save_figure(fig, base / 'outputs/plots/monthly_reports.png')
    seriousness = pd.Series({c.removeprefix('Seriousness_'): int(df[c + '_Binary'].eq(1).sum()) for c in SERIOUS})
    bar(seriousness, 'Flagged serious outcomes (may overlap)', 'seriousness_counts.png', base)
    death = df.loc[df[TARGET].eq(1)]
    for col, stem in [('Drug_Brand_Name', 'drugs'), ('Reaction_0', 'reactions')]:
        bar(known_counts(death, col), f'Top {stem} in death-associated reports', f'top_death_{stem}.png', base)
    for country, aliases in {'USA': ['US', 'USA', 'UNITED STATES'], 'Canada': ['CA', 'CANADA'], 'Brazil': ['BR', 'BRAZIL']}.items():
        subset = df.loc[df['Country'].str.upper().isin(aliases)]
        for col, stem in [('Drug_Brand_Name', 'drugs'), ('Reaction_0', 'reactions')]:
            bar(known_counts(subset, col), f'{country}: top reported {stem}', f'top_{stem}_{country.lower()}.png', base)
    valid = df.loc[df['Drug_Brand_Name'].ne('Unknown') & df['Reaction_0'].ne('Unknown')]
    drugs, reactions = known_counts(valid, 'Drug_Brand_Name').index, known_counts(valid, 'Reaction_0').index
    selected = valid.loc[valid['Drug_Brand_Name'].isin(drugs) & valid['Reaction_0'].isin(reactions)]
    heat = pd.crosstab(selected['Drug_Brand_Name'], selected['Reaction_0']).reindex(index=drugs, columns=reactions, fill_value=0)
    heat.to_csv(base / 'outputs/results/drug_reaction_cooccurrence.csv')
    fig, ax = plt.subplots(figsize=(14, 10))
    if heat.size:
        im = ax.imshow(heat.values, aspect='auto', cmap='Blues')
        fig.colorbar(im, ax=ax, label='Co-occurrence count')
        ax.set_xticks(range(len(reactions)), reactions, rotation=55, ha='right')
        ax.set_yticks(range(len(drugs)), drugs)
    ax.set_title('Top drugs and primary reactions: report co-occurrence')
    save_figure(fig, base / 'outputs/plots/drug_reaction_heatmap.png')
    bar(df[TARGET].map({0: 'Non-death', 1: 'Death'}).fillna('Unknown').value_counts(),
        'Death outcome coding', 'death_outcome_distribution.png', base)
    print('EDA saved.', flush=True)


class ReportPreparer(BaseEstimator, TransformerMixin):
    """Learn numeric medians and rare categories exclusively from training rows."""
    def __init__(self, rare_threshold=25):
        self.rare_threshold = rare_threshold

    def fit(self, X, y=None):
        self.medians_ = X[NUM].apply(pd.to_numeric, errors='coerce').median().fillna(0)
        self.frequent_ = {c: set(X[c].fillna('Unknown').astype(str).value_counts().loc[lambda s: s >= self.rare_threshold].index) for c in CAT}
        return self

    def transform(self, X):
        result = X[FEATURES].copy()
        result[NUM] = result[NUM].apply(pd.to_numeric, errors='coerce').fillna(self.medians_)
        for c in CAT:
            text = result[c].fillna('Unknown').astype(str)
            result[c] = text.where(text.isin(self.frequent_[c]), '__RARE__')
        return result


def encoder():
    return ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scaler', StandardScaler())]), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32), CAT),
    ], sparse_threshold=1.0)


def algorithms(seed=42):
    return {
        'Logistic Regression': LogisticRegression(max_iter=500, solver='saga', random_state=seed),
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=25, random_state=seed, n_jobs=-1),
        'Linear SVM': LinearSVC(random_state=seed, max_iter=5000, dual='auto'),
        'KNN': KNeighborsClassifier(n_neighbors=5, algorithm='brute', n_jobs=-1),
    }


def subset(df, size, seed):
    model_df = df.loc[df[TARGET].notna()].copy()
    model_df[TARGET] = model_df[TARGET].astype(int)
    if model_df[TARGET].nunique() != 2:
        raise ValueError('Supervised training needs both known death and non-death outcomes.')
    if size < 20:
        raise ValueError('Use a sample size of at least 20.')
    if len(model_df) > size:
        model_df, _ = train_test_split(model_df, train_size=size, stratify=model_df[TARGET], random_state=seed)
    if model_df[TARGET].value_counts().min() < 3:
        raise ValueError('Too few examples in the minority class; increase sample size.')
    return model_df


def train(df, run, size, base=BASE, seed=42, rare_threshold=25):
    model_df = subset(df, size, seed)
    X_train, X_test, y_train, y_test = train_test_split(model_df[FEATURES], model_df[TARGET],
        test_size=.2, stratify=model_df[TARGET], random_state=seed)
    # Persist source row positions to audit train/test disjointness. No report identifiers are available.
    split = pd.DataFrame({'source_row': list(X_train.index) + list(X_test.index),
                           'partition': ['train'] * len(X_train) + ['test'] * len(X_test)})
    split.to_csv(base / f'outputs/results/{run.lower()}_split_rows.csv', index=False)
    prepare = ReportPreparer(rare_threshold).fit(X_train)
    training, testing = prepare.transform(X_train), prepare.transform(X_test)
    before = y_train.value_counts().sort_index().to_dict()
    resample_seconds = 0.0
    numeric_scale = None
    if run == 'SMOTENC':
        minority = int(y_train.value_counts().min())
        if minority < 2:
            raise ValueError('SMOTENC requires at least two minority training examples.')
        # Standardize continuous variables before nearest-neighbor distances.
        numeric_scale = StandardScaler().fit(training[NUM])
        training[NUM] = numeric_scale.transform(training[NUM])
        testing[NUM] = numeric_scale.transform(testing[NUM])
        sampler = SMOTENC(categorical_features=CAT, random_state=seed, k_neighbors=min(5, minority - 1))
        start = time.perf_counter()
        training, y_train = sampler.fit_resample(training, y_train)
        resample_seconds = time.perf_counter() - start
    after = pd.Series(y_train).value_counts().sort_index().to_dict()
    enc = encoder()
    start = time.perf_counter()
    encoded_train = enc.fit_transform(training)
    encoded_test = enc.transform(testing)
    preprocessing_seconds = time.perf_counter() - start
    results = []
    reports, manifest = {}, {}
    for name, model in {'Majority Dummy': DummyClassifier(strategy='most_frequent'), **algorithms(seed)}.items():
        print(f'{run}: fitting {name} on {encoded_train.shape}', flush=True)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(encoded_train, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(encoded_test)
        predict_seconds = time.perf_counter() - start
        scores = model.predict_proba(encoded_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(encoded_test)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        result = dict(run=run, model=name, accuracy=accuracy_score(y_test, pred),
            balanced_accuracy=balanced_accuracy_score(y_test, pred), precision_death=precision_score(y_test, pred, zero_division=0),
            recall_death=recall_score(y_test, pred, zero_division=0), f1_death=f1_score(y_test, pred, zero_division=0),
            roc_auc=roc_auc_score(y_test, scores), average_precision=average_precision_score(y_test, scores),
            train_seconds=fit_seconds, predict_seconds=predict_seconds, preprocessing_seconds=preprocessing_seconds,
            resample_seconds=resample_seconds, sample_rows=len(model_df), train_rows=len(y_train), test_rows=len(y_test),
            encoded_features=encoded_train.shape[1], tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught))
        results.append(result)
        stem = name.lower().replace(' ', '_')
        reports[name] = classification_report(y_test, pred, output_dict=True, zero_division=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=['Non-death', 'Death']).plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
        ax.set_title(f'{run}: {name}')
        save_figure(fig, base / f'outputs/plots/{run.lower()}_{stem}_confusion_matrix.png')
        pd.DataFrame({'source_row': X_test.index, 'actual': y_test.to_numpy(), 'predicted': pred,
                      'score': scores}).to_csv(base / f'outputs/results/{run.lower()}_{stem}_predictions.csv', index=False)
        if name != 'Majority Dummy':
            joblib.dump({'prepare': prepare, 'numeric_scale': numeric_scale, 'encoder': enc,
                         'model': model, 'features': FEATURES, 'run': run}, base / f'models/{run.lower()}_{stem}.joblib', compress=3)
        manifest[name] = [str(w.message) for w in caught]
        print(f'{name}: accuracy={result["accuracy"]:.4f}, recall={result["recall_death"]:.4f}, F1={result["f1_death"]:.4f}', flush=True)
        pd.DataFrame(results).to_csv(base / f'outputs/results/{run.lower()}_metrics.csv', index=False)
    write_json(base / f'outputs/results/{run.lower()}_classification_reports.json', reports)
    write_json(base / f'outputs/results/{run.lower()}_run.json', dict(seed=seed, requested_sample=size,
        actual_sample=len(model_df), test_fraction=.2, rare_threshold=rare_threshold,
        train_before_resampling=before, train_after_resampling=after, test_counts=y_test.value_counts().to_dict(),
        warnings=manifest, algorithms={k: v.get_params() for k, v in algorithms(seed).items()}))
    return pd.DataFrame(results)


def summary(raw, clean, base=BASE):
    values = {'source_rows': len(raw), 'source_columns': len(raw.columns), 'known_death_outcomes': int(clean[TARGET].notna().sum()),
        'death_reports': int(clean[TARGET].eq(1).sum()), 'non_death_reports': int(clean[TARGET].eq(0).sum()),
        'unknown_death_outcomes': int(clean[TARGET].isna().sum()), 'exact_duplicate_source_rows': int(raw.duplicated().sum()),
        'hospitalization_flagged': int(clean['Seriousness_Hospitalization_Binary'].eq(1).sum()),
        'disabling_flagged': int(clean['Seriousness_Disabling_Binary'].eq(1).sum()),
        'invalid_or_missing_weight': int(clean['Weight'].isna().sum()),
        'invalid_or_missing_date': int(clean['Transmission_Date'].isna().sum())}
    pd.Series(values).rename_axis('metric').to_csv(base / 'outputs/results/dataset_summary.csv', header=['value'])
    clean[TARGET].value_counts(dropna=False).rename_axis('death_binary').to_csv(base / 'outputs/results/class_distribution.csv', header=['reports'])
    return values


def compare(base=BASE):
    result = base / 'outputs/results'
    paths = [result / 'baseline_metrics.csv', result / 'smotenc_metrics.csv']
    if not all(p.exists() for p in paths):
        raise FileNotFoundError('Run both baseline and SMOTENC training before building comparisons.')
    comparison = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    comparison.to_csv(result / 'model_comparison.csv', index=False)
    for metric, title, stem in [('accuracy', 'Accuracy', 'model_accuracy'), ('recall_death', 'Death recall', 'death_recall'),
                                 ('f1_death', 'Death F1', 'death_f1'), ('precision_death', 'Death precision', 'death_precision')]:
        pivot = comparison.loc[comparison.model.ne('Majority Dummy')].pivot(index='model', columns='run', values=metric)
        ax = pivot.plot.bar(figsize=(10, 6), color=['#2563a6', '#d98232'])
        ax.set(title=f'{title}: baseline and SMOTENC (different sample sizes)', ylim=(0, 1.05), ylabel=title, xlabel='')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        save_figure(ax.figure, base / f'outputs/plots/{stem}_comparison.png')
    dataset = pd.read_csv(result / 'dataset_summary.csv')
    rows = ['# Results', '', 'Measured from the supplied merged CSV. Historical guide figures are recorded separately in presentation/README.md.', '',
        '## Dataset', '', '| Metric | Value |', '| --- | ---: |']
    rows += [f'| {r.metric} | {int(r.value):,} |' for r in dataset.itertuples()]
    rows += ['', '## Evaluation', '', '| Run | Model | Accuracy | Death precision | Death recall | Death F1 | Average precision | Fit seconds |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in comparison.itertuples():
        rows.append(f'| {r.run} | {r.model} | {r.accuracy:.4f} | {r.precision_death:.4f} | {r.recall_death:.4f} | {r.f1_death:.4f} | {r.average_precision:.4f} | {r.train_seconds:.2f} |')
    best = comparison.loc[(comparison.run == 'Baseline') & comparison.model.ne('Majority Dummy')].sort_values('f1_death', ascending=False).iloc[0]
    rows += ['', '## Measured observations', '',
        f'{best.model} has the highest baseline death F1 in this run ({best.f1_death:.4f}), with accuracy {best.accuracy:.2%}, death precision {best.precision_death:.2%}, and death recall {best.recall_death:.2%}. This ranking describes this particular split and is not a clinical model recommendation.', '']
    for name in algorithms():
        pair = comparison.loc[comparison.model.eq(name)].set_index('run')
        delta = pair.loc['SMOTENC', 'recall_death'] - pair.loc['Baseline', 'recall_death']
        rows.append(f'{name}: death recall changes by {delta * 100:+.2f} percentage points between the two runs. Baseline F1 is {pair.loc["Baseline", "f1_death"]:.4f}; SMOTENC F1 is {pair.loc["SMOTENC", "f1_death"]:.4f}.')
        rows.append('')
    rows += ['', '## Interpretation and limits', '',
        'Baseline uses up to 65,000 known-target rows; SMOTENC uses up to 15,000 before resampling. Each run has its own untouched stratified 20% test split. Differences combine sample-size effects with balancing effects, so this is not a controlled estimate of the effect of SMOTENC. For a controlled comparison, rerun both with the same sample size and seed.', '',
        'Rare categories, imputation, scaling and encoding are fitted on training rows only. SMOTENC sees training rows only. Numeric fields are standardized before its neighbor search. Hospitalization, disability and death codes are excluded from predictors.', '',
        'These are retrospective classifications of reports with known outcome codes, not prospective predictions of patient mortality or evidence that a drug caused death. Reactions may already describe an outcome. Spontaneous report counts are not drug incidence rates. The merged file has no report identifier: repeated reports, follow-up versions and shared patients cannot reliably be grouped, and overlapping records may inflate random-split performance. Unknown outcomes are excluded and may introduce selection bias. No clinical validation, confidence intervals or external test cohort is provided.', '',
        'Exact source duplicates are counted but retained to preserve the documented source experiment. Review deduplication and report-level identifiers before making substantive conclusions. All convergence warnings are saved in each run JSON; warning flags also appear in metrics CSVs. Models are trusted local joblib artifacts; do not load untrusted joblib files.', '']
    (base / 'RESULTS.md').write_text('\n'.join(rows), encoding='utf-8')
    return comparison


def synthetic(raw, output, target_bytes, chunk_rows=100000, seed=42):
    if target_bytes <= 0 or chunk_rows <= 0:
        raise ValueError('Target bytes and chunk rows must be positive.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to append to an existing experiment: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    rng, total = np.random.default_rng(seed), 0
    while not output.exists() or output.stat().st_size < target_bytes:
        chunk = raw.iloc[rng.integers(0, len(raw), size=chunk_rows)].reset_index(drop=True).copy()
        weight = pd.to_numeric(chunk['Weight'], errors='coerce').astype(float)
        valid = weight.between(0, 650, inclusive='right')
        weight.loc[valid] = (weight.loc[valid] + rng.normal(0, 1, int(valid.sum()))).clip(.1, 650)
        chunk['Weight'] = weight
        chunk.to_csv(output, mode='a', index=False, header=not output.exists())
        total += len(chunk)
    write_json(output.with_suffix('.manifest.json'), {'purpose': 'Bootstrapped scalability data, not additional real reports',
        'rows': total, 'bytes': output.stat().st_size, 'target_bytes': target_bytes, 'seed': seed})
    return total


def provenance(path, args, base=BASE):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    write_json(base / 'outputs/results/provenance.json', {'input_filename': Path(path).name,
        'input_sha256': digest.hexdigest(), 'input_bytes': Path(path).stat().st_size,
        'python': platform.python_version(), 'packages': {p: importlib.metadata.version(p)
        for p in ['pandas', 'numpy', 'matplotlib', 'scikit-learn', 'imbalanced-learn', 'joblib']},
        'seed': args.seed, 'baseline_sample': args.baseline_sample, 'smotenc_sample': args.smotenc_sample,
        'input_kind': 'User-supplied merged adverse-event CSV; not fetched directly from OpenFDA'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['all', 'inspect', 'clean', 'eda', 'baseline', 'smotenc', 'synthetic', 'outputs'])
    parser.add_argument('--data', type=Path, default=BASE / 'data/raw/2025_merged.csv')
    parser.add_argument('--baseline-sample', type=int, default=65000)
    parser.add_argument('--smotenc-sample', type=int, default=15000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rare-threshold', type=int, default=25)
    parser.add_argument('--synthetic-bytes', type=int, default=1000000000)
    parser.add_argument('--chunk-rows', type=int, default=100000)
    parser.add_argument('--synthetic-output', type=Path, default=BASE / 'data/synthetic/synthetic_drug_event_1GB.csv')
    args = parser.parse_args(argv)
    folders()
    if args.step == 'outputs':
        compare()
        return
    raw = read_raw(args.data)
    if args.step == 'synthetic':
        print(f'Generated {synthetic(raw, args.synthetic_output, args.synthetic_bytes, args.chunk_rows, args.seed):,} synthetic rows.')
        return
    if args.step in ['all', 'inspect']:
        inspect(raw)
    clean = clean_data(raw)
    summary(raw, clean)
    if args.step in ['all', 'clean']:
        clean.to_csv(BASE / 'data/processed/2025_merged_clean.csv', index=False)
        raw.sample(n=min(200, len(raw)), random_state=args.seed).to_csv(BASE / 'data/sample_patient_drug_events.csv', index=False)
        print('Cleaned data saved.', flush=True)
    if args.step in ['all', 'eda']:
        eda(clean)
    if args.step in ['all', 'baseline']:
        train(clean, 'Baseline', args.baseline_sample, seed=args.seed, rare_threshold=args.rare_threshold)
    if args.step in ['all', 'smotenc']:
        train(clean, 'SMOTENC', args.smotenc_sample, seed=args.seed, rare_threshold=args.rare_threshold)
    if args.step == 'all':
        compare()
        provenance(args.data, args)
    print(f'Finished {args.step}. Outputs: {BASE / "outputs"}', flush=True)


if __name__ == '__main__':
    # Import under the canonical package name so saved estimators can be loaded
    # by a different process (pickle must not refer to __main__.ReportPreparer).
    from src.project import main as cli_main
    cli_main()
