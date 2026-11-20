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

The scripts reproduce the notebook workflow with four final strategies:

- VIF + stable LASSO selection + logistic regression;
- Elastic Net selection + RBF SVM;
- Random Forest tuning + Gini/RFECV selection;
- Random Forest tuning + permutation-importance selection.

`train.py --pipeline all` can take a long time because the Random Forest grid search and RFECV perform many fits. Each pipeline can be run separately. Generated datasets, models, metrics and figures are ignored by Git; the original notebook and raw CSV files are never modified.
