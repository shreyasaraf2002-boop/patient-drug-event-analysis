"""Generate a standalone Colab notebook with the shared source embedded."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.splitlines(keepends=True)})


md('# Patient Drug Event Analysis\n\nRun cells in order. This is retrospective report classification using your merged CSV. The notebook includes the complete source and requires no repository clone. Unknown outcome codes are excluded from training. Measured results are produced from your uploaded data; the guide\'s historical scores are not copied into the evaluation.')
code('%pip -q install pandas==2.2.3 numpy==2.2.6 matplotlib==3.10.3 scikit-learn==1.6.1 imbalanced-learn==0.13.0 joblib==1.5.1')
md('## Create the project\n\nThis cell writes the same shared Python module used locally. Restart the runtime after package installation if prompted, then continue.')
sources = {f'src/{name}': (BASE / 'src' / name).read_text(encoding='utf-8') for name in ['__init__.py', 'project.py', 'predict.py']}
code('from pathlib import Path\nimport sys, json\nBASE = Path("/content/patient-drug-event-analysis")\nBASE.mkdir(parents=True, exist_ok=True)\nSOURCES = ' + repr(sources) + '\nfor name, source in SOURCES.items():\n    path = BASE / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\nsys.path.insert(0, str(BASE))\nfrom src.project import *\nfolders(BASE)\nprint(BASE)')
md('## Upload the real merged CSV\n\nSelect exactly one source CSV. For Google Drive, replace this cell with a Drive mount and set DATA_PATH to your file.')
code('from google.colab import files\nuploaded = files.upload()\nif len(uploaded) != 1:\n    raise ValueError("Upload exactly one merged CSV")\nname = next(iter(uploaded))\nDATA_PATH = BASE / "data/raw/2025_merged.csv"\nDATA_PATH.write_bytes(uploaded[name])\ndel uploaded\nraw = read_raw(DATA_PATH)\ninspect(raw, BASE)\ndisplay(raw.head())\ndisplay(pd.read_csv(BASE / "outputs/results/source_column_profile.csv"))')
md('## Clean and summarize\n\nWeight outside (0,650] becomes missing. Invalid dates become missing. Only death codes 1 and 2 become known targets; no Age column is invented.')
code('clean = clean_data(raw)\nclean.to_csv(BASE / "data/processed/2025_merged_clean.csv", index=False)\nraw.sample(min(200, len(raw)), random_state=42).to_csv(BASE / "data/sample_patient_drug_events.csv", index=False)\ndisplay(pd.Series(summary(raw, clean, BASE)).to_frame("value"))')
md('## Exploratory analysis\n\nReport counts and co-occurrence describe reporting patterns; they do not establish drug risk or causation.')
code('eda(clean, BASE)\nfrom IPython.display import Image, display\nfor name in ["top_countries.png", "drug_reaction_heatmap.png", "death_outcome_distribution.png"]:\n    display(Image(filename=str(BASE / "outputs/plots" / name)))')
md('## Baseline models\n\nTraining-only category grouping, median imputation, scaling and encoding. The dummy reference helps interpret accuracy. KNN may take several minutes.')
code('MODEL_SAMPLE_SIZE = 65000\nSMOTENC_SAMPLE_SIZE = 15000\nSEED = 42\nbaseline_results = train(clean, "Baseline", MODEL_SAMPLE_SIZE, BASE, SEED)\ndisplay(baseline_results)')
md('## Training-only SMOTENC\n\nThe original test split remains unchanged. Continuous fields are standardized before neighbor search. This smaller sample follows the guide; use the same sample size in both runs for a controlled comparison.')
code('smote_results = train(clean, "SMOTENC", SMOTENC_SAMPLE_SIZE, BASE, SEED)\ndisplay(smote_results)')
md('## Compare and save\n\nRead generated RESULTS.md for measured metrics and limitations. Random splits may share related or duplicate reports because no report identifier is available.')
code('comparison = compare(BASE)\nfrom types import SimpleNamespace\nprovenance(DATA_PATH, SimpleNamespace(seed=SEED, baseline_sample=MODEL_SAMPLE_SIZE, smotenc_sample=SMOTENC_SAMPLE_SIZE), BASE)\ndisplay(comparison)\nfor name in ["model_accuracy_comparison.png", "death_recall_comparison.png", "death_f1_comparison.png"]:\n    display(Image(filename=str(BASE / "outputs/plots" / name)))\nprint((BASE / "RESULTS.md").read_text())')
md('## Optional scalability experiment\n\nDisabled by default. This bootstraps whole rows and modifies only weight; it creates no new real clinical evidence.')
code('RUN_SYNTHETIC_EXPERIMENT = False\nif RUN_SYNTHETIC_EXPERIMENT:\n    synthetic(raw, BASE / "data/synthetic/synthetic_drug_event_1GB.csv", 1_000_000_000)')
md('## Download results\n\nThe archive includes results, plots, notebook-source modules and trusted model bundles. Full raw, processed and synthetic data are excluded.')
code('import zipfile\nZIP_PATH = Path("/content/patient-drug-event-analysis-results.zip")\nwith zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:\n    for folder in ["outputs", "models", "src"]:\n        for path in (BASE / folder).rglob("*"):\n            if path.is_file() and "__pycache__" not in path.parts:\n                archive.write(path, path.relative_to(BASE))\n    archive.write(BASE / "RESULTS.md", "RESULTS.md")\nfiles.download(str(ZIP_PATH))')
notebook = {'cells': cells, 'metadata': {'colab': {'name': 'Patient_Drug_Event_Analysis_Colab.ipynb'},
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
for index, cell in enumerate(cells):
    cell['id'] = f'cell-{index:03}'
path = BASE / 'notebooks/Patient_Drug_Event_Analysis_Colab.ipynb'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(notebook, indent=2), encoding='utf-8')
print(path)
