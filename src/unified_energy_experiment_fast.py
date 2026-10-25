"""
Unified, reproducible experiment for the UCI Energy Efficiency dataset.
Place this script in the same folder as ENB2012_data.xlsx, then run:
    python unified_energy_experiment.py

Outputs:
- unified_split_results.csv
- unified_summary.csv
- unified_mi_selected_features.csv
- unified_stability_summary.csv
- unified_preprocessing_report.txt
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "ENB2012_data.xlsx"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTER_SEEDS = list(range(100, 130))
TEST_SIZE = 0.20
INNER_FOLDS = 3
FEATURES = [f"X{i}" for i in range(1, 9)]
TARGETS = {"Y1": "Heating Load", "Y2": "Cooling Load"}

SCENARIOS = {
    "All 8 features": FEATURES,
    "Geometry only (X1-X5)": FEATURES[:5],
    "Plus orientation (X1-X6)": FEATURES[:6],
    "Without glazing distribution (X1-X7)": FEATURES[:7],
    "Exclude X6 (orientation)": [x for x in FEATURES if x != "X6"],
    "Exclude X8 (glazing distribution)": [x for x in FEATURES if x != "X8"],
    "Exclude X6 and X8": [x for x in FEATURES if x not in ("X6", "X8")],
}

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE.name}. Put the Excel dataset at data/ENB2012_data.xlsx."
        )
    raw = pd.read_excel(DATA_FILE)
    raw.columns = raw.columns.astype(str).str.strip()
    required = FEATURES + list(TARGETS.keys())
    missing_columns = [c for c in required if c not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = raw[required].copy()
    for c in required:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    missing_counts = data.isna().sum()
    duplicate_count = int(data.duplicated().sum())
    report = [
        f"Dataset file: {DATA_FILE.name}",
        f"Rows: {len(data)}",
        f"Required columns: {len(required)}",
        f"Missing/non-numeric values by column: {missing_counts.to_dict()}",
        f"Exact duplicate rows across required columns: {duplicate_count}",
        f"Feature ranges: {data[FEATURES].agg(['min','max']).to_dict()}",
        f"Target ranges: {data[list(TARGETS)].agg(['min','max']).to_dict()}",
        f"Outer evaluation: {len(OUTER_SEEDS)} independent seeded 80/20 random partitions (seeds 100-129).",
        f"MI selection: SelectKBest with k in [3, 5, 6, 7, 8] selected by {INNER_FOLDS}-fold CV using only the training partition.",
        "Scaling: StandardScaler inside Linear Regression pipelines only; Random Forest uses unscaled inputs.",
        "No imputation, row deletion, or outlier removal is applied automatically."
    ]
    if missing_counts.any():
        (OUT_DIR / "unified_preprocessing_report.txt").write_text("\n".join(report), encoding="utf-8")
        raise ValueError("Missing/non-numeric values found; inspect unified_preprocessing_report.txt.")
    (OUT_DIR / "unified_preprocessing_report.txt").write_text("\n".join(report), encoding="utf-8")

    # MI treats X6 orientation and X8 glazing distribution as discrete-valued features.
    discrete_mask = np.array([False, False, False, False, False, True, False, True])

    def mi_score(X, y):
        # X remains the original 8-column matrix until SelectKBest applies the selected mask.
        return mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)

    rows = []
    selected_rows = []
    for split_number, seed in enumerate(OUTER_SEEDS, start=1):
        print(f"Running split {split_number}/{len(OUTER_SEEDS)} (seed={seed})...", flush=True)
        train_idx, test_idx = train_test_split(
            np.arange(len(data)), test_size=TEST_SIZE, random_state=seed
        )
        for target, target_name in TARGETS.items():
            y_train = data.iloc[train_idx][target]
            y_test = data.iloc[test_idx][target]
            for model_name in ("Linear Regression", "Random Forest"):
                # Evaluate each predefined feature scenario on identical outer splits.
                for scenario, cols in SCENARIOS.items():
                    X_train = data.iloc[train_idx][cols]
                    X_test = data.iloc[test_idx][cols]
                    if model_name == "Linear Regression":
                        model = Pipeline([
                            ("scale", StandardScaler()),
                            ("model", LinearRegression())
                        ])
                    else:
                        model = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=1)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    rows.append({
                        "seed": seed, "target": target, "target_name": target_name,
                        "model": model_name, "method": "Predefined/targeted scenario",
                        "scenario": scenario, "feature_count": len(cols),
                        "features": ",".join(cols),
                        "MAE": mean_absolute_error(y_test, pred),
                        "RMSE": mean_squared_error(y_test, pred) ** 0.5
                    })

                # MI selection is fully inside training data with inner CV.
                if model_name == "Linear Regression":
                    pipe = Pipeline([
                        ("scale", StandardScaler()),
                        ("select", SelectKBest(score_func=mi_score)),
                        ("model", LinearRegression())
                    ])
                else:
                    pipe = Pipeline([
                        ("select", SelectKBest(score_func=mi_score)),
                        ("model", RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=1))
                    ])
                cv = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
                search = GridSearchCV(
                    pipe, param_grid={"select__k": [3, 5, 6, 7, 8]},
                    scoring="neg_mean_absolute_error", cv=cv, n_jobs=1, refit=True
                )
                X_train_all = data.iloc[train_idx][FEATURES]
                X_test_all = data.iloc[test_idx][FEATURES]
                search.fit(X_train_all, y_train)
                pred = search.predict(X_test_all)
                chosen_k = int(search.best_params_["select__k"])
                support = search.best_estimator_.named_steps["select"].get_support()
                chosen = [f for f, keep in zip(FEATURES, support) if keep]
                rows.append({
                    "seed": seed, "target": target, "target_name": target_name,
                    "model": model_name, "method": "MI + inner CV",
                    "scenario": f"MI-selected ({chosen_k} features)", "feature_count": chosen_k,
                    "features": ",".join(chosen),
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": mean_squared_error(y_test, pred) ** 0.5
                })
                selected_rows.append({
                    "seed": seed, "target": target, "target_name": target_name,
                    "model": model_name, "selected_k": chosen_k, "selected_features": ",".join(chosen)
                })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "unified_split_results.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(OUT_DIR / "unified_mi_selected_features.csv", index=False)

    # Paired comparisons against all-8 baseline on the same seed.
    baseline = results[results["scenario"] == "All 8 features"].copy()
    summaries = []
    for (target, model, method, scenario, feature_count), group in results.groupby(
        ["target", "model", "method", "scenario", "feature_count"], dropna=False
    ):
        base = baseline[(baseline.target == target) & (baseline.model == model)].set_index("seed")
        comp = group.set_index("seed").reindex(base.index)
        for metric in ("MAE", "RMSE"):
            delta = comp[metric] - base[metric]
            summaries.append({
                "target": target, "target_name": TARGETS[target], "model": model,
                "method": method, "scenario": scenario, "feature_count_mean": float(group.feature_count.mean()),
                "metric": metric, "mean_score": comp[metric].mean(), "std_score": comp[metric].std(),
                "all_8_baseline_mean": base[metric].mean(),
                "mean_difference_vs_baseline": delta.mean(),
                "percent_change_vs_baseline": delta.mean() / base[metric].mean() * 100,
                "splits_better_than_baseline": int((delta < 0).sum()),
                "splits_worse_than_baseline": int((delta > 0).sum()),
                "splits_tied": int((delta == 0).sum()),
                "n_splits": int(comp[metric].notna().sum())
            })
    pd.DataFrame(summaries).to_csv(OUT_DIR / "unified_summary.csv", index=False)

    # Jaccard stability of the MI-selected subsets.
    stability = []
    sel_df = pd.DataFrame(selected_rows)
    for (target, model), group in sel_df.groupby(["target", "model"]):
        sets = [set(s.split(",")) if s else set() for s in group["selected_features"]]
        scores = [
            len(a & b) / len(a | b) if len(a | b) else 1.0
            for a, b in itertools.combinations(sets, 2)
        ]
        feature_freq = {
            f: sum(f in s for s in sets) / len(sets) * 100 for f in FEATURES
        }
        stability.append({
            "target": target, "target_name": TARGETS[target], "model": model,
            "n_splits": len(sets), "mean_pairwise_jaccard": float(np.mean(scores)),
            "min_pairwise_jaccard": float(np.min(scores)),
            "max_pairwise_jaccard": float(np.max(scores)),
            "mean_selected_feature_count": float(np.mean([len(s) for s in sets])),
            **{f"{f}_selection_percent": feature_freq[f] for f in FEATURES}
        })
    pd.DataFrame(stability).to_csv(OUT_DIR / "unified_stability_summary.csv", index=False)
    print("Completed unified experiment.")
    print("Files written: unified_split_results.csv, unified_summary.csv,")
    print("unified_mi_selected_features.csv, unified_stability_summary.csv,")
    print("unified_preprocessing_report.txt")

if __name__ == "__main__":
    main()
