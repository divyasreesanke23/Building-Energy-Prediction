# Building Energy Prediction: Feature Scenarios, Accuracy, and Selection Stability

A reproducible research project comparing Linear Regression (LR) and Random Forest (RF) for heating-load (Y1) and cooling-load (Y2) prediction on the UCI Energy Efficiency dataset. The experiment evaluates seven predefined feature scenarios and Mutual Information (MI)-based selection, using prediction error and selection consistency across repeated splits.

## Current experiment

- Dataset: UCI Energy Efficiency, 768 rows, predictors X1–X8, targets Y1 and Y2.
- Models: Linear Regression and Random Forest (`n_estimators=40`, `random_state=42`, `n_jobs=1`).
- Outer evaluation: 30 seeded 80/20 random train–test splits (seeds 100–129).
- MI selection: `SelectKBest(mutual_info_regression)` with candidate k values `{3, 5, 6, 7, 8}` and shuffled 3-fold inner cross-validation on training data only.
- Metrics: MAE, RMSE, and mean pairwise Jaccard similarity of selected feature sets.
- LR uses `StandardScaler` inside a pipeline; RF inputs are not scaled.
- No imputation, row deletion, or outlier removal is applied.

**Scope note:** the repeated random splits are resampling evaluations on one simulated benchmark, not independent external datasets. The dataset file does not provide an explicit building-shape ID in the X1–X8 columns, so this repository does not claim a valid leave-one-shape-out experiment.

## Dataset download

Download the `ENB2012_data.xlsx` file from the official [UCI Energy Efficiency dataset page](https://archive.ics.uci.edu/dataset/242/energy%2Befficiency) and place it at:

`data/ENB2012_data.xlsx`

The dataset is not committed in this repository; see the UCI page for source, citation, and license details.

## Run on Windows (VS Code / PowerShell)

From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/unified_energy_experiment_fast.py
```

On macOS/Linux, use `python3 -m venv .venv` and `source .venv/bin/activate`.

The script writes generated CSV and preprocessing-report files to `results/`. The saved CSVs in this repository were derived from the recorded 30-split experiment output. Re-run the script before replacing them; a full run can take several minutes depending on your computer.

## Repository contents

- `src/unified_energy_experiment_fast.py` — main experiment script used for the reported 30-split, 40-tree setup.
- `results/unified_split_results.csv` — per-split predictions metrics used in the paper's analyses.
- `figures/Figure1_MAE_Comparison.png` — MAE distribution comparison for all features versus MI-selected features.
- `figures/Figure2_Feature_Selection_Frequency.png` — feature selection frequency across repeated splits.
- `requirements.txt` — minimum Python package requirements (not a record of the original execution environment).

## Main findings from the saved results

Mean MAE with all eight features:

| Target | Linear Regression | Random Forest |
|---|---:|---:|
| Heating load (Y1) | 2.0533 | 0.3250 |
| Cooling load (Y2) | 2.2483 | 0.9911 |

The saved results show lower mean MAE for RF than LR on this benchmark and protocol. MI selection did not consistently improve error over the all-feature baseline for every target/model combination. Conclusions should remain limited to this dataset and evaluation design.

## Reproducibility notes

The script records seeds and settings used for the experiment. The original run log did not record the execution date or installed package versions; this repository therefore does not claim exact original environment versions. The requirements file lists minimum compatible dependencies for rerunning the code.

## Citation

Please cite the UCI Energy Efficiency dataset and the original study:

A. Tsanas and A. Xifara, “Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools,” *Energy and Buildings*, vol. 49, pp. 560–567, 2012, doi: 10.1016/j.enbuild.2012.03.003.

UCI Machine Learning Repository, “Energy Efficiency,” doi: 10.24432/C51307.
