"""Bundle the complete code and measured outputs without large datasets or venv."""
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / 'Patient_Drug_Event_Analysis_Complete.zip'
files = []
for name in ['README.md', 'RESULTS.md', 'VALIDATION.md', 'requirements.txt', 'requirements-lock.txt', '.gitignore']:
    files.append(BASE / name)
for folder in ['src', 'notebooks', 'tests', 'tools', 'presentation', 'models', 'outputs/plots', 'outputs/results']:
    files.extend(p for p in (BASE / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend([BASE / 'data/README.md', BASE / 'data/sample_patient_drug_events.csv'])
with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(BASE).as_posix())
with zipfile.ZipFile(OUTPUT) as archive:
    assert archive.testzip() is None
    assert not any(name.startswith(('data/raw/', 'data/processed/', 'data/synthetic/', '.venv/')) for name in archive.namelist())
print(f'{OUTPUT}\n{len(files)} files; {OUTPUT.stat().st_size:,} bytes')
