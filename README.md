# Heart Failure Readmission

Reproducible analysis for predicting readmission within six months after hospital discharge in patients with heart failure. 

## Project layout

- `src/heart_failure/`: reusable data, preprocessing, modelling, evaluation and plotting code.
- `scripts/`: small command-line entry points which orchestrate the package.
- `configs/default.yaml`: paths, random seed and modelling options.
- `data/`: raw, intermediate and processed data (not committed).
- `models/`: trained model artefacts (not committed).
- `reports/`: exported metrics and figures.
- `tests/`: automated checks for transformations.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Workflow

Copy the source CSV files into `data/raw/` using the names configured in `configs/default.yaml`, then run:

```powershell
python scripts/make_dataset.py
python scripts/train.py --pipeline all
python scripts/evaluate.py --pipeline all
```

The clinical preprocessing rules still need to be transferred from the notebook into `src/heart_failure/cleaning.py`, one verified block at a time. This scaffold deliberately does not modify the notebook or the existing CSV files.
