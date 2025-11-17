# -*- coding: utf-8 -*-
"""
Composting parameter tuner with RandomForest + SHAP + Genetic Algorithm (GA).
- Reads data from 总表.csv (must contain target column 'GI')
- Trains RandomForest, evaluates on a holdout set
- Computes SHAP values, exports tables & plots
- Loads fixed parameters from fixed_params.csv (Feature,Value)
- Runs a GA to maximize predicted GI by tuning remaining features within data-driven bounds
- Exports candidates and best suggestion to output/ directory
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

# =========================
# Global plot settings
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.title_fontsize'] = 20

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================
# Paths
# =========================
INPUT_FILE = "总表.csv"
FIXED_PARAMS_FILE = "fixed_params.csv"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("Input file not found: 总表.csv")

# =========================
# Step 1: Load & clean data
# =========================
df = pd.read_csv(INPUT_FILE, na_values=["#VALUE!", "NA", "NaN", "null", ""])
if "GI" not in df.columns:
    raise KeyError("Target column 'GI' not found in the input data.")

df = df.dropna(subset=["GI"]).copy()
X = df.drop(columns=["GI"])
y = df["GI"].astype(float)

num_imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(num_imputer.fit_transform(X), columns=X.columns, index=X.index)

# =========================
# Step 2: Train RandomForest
# =========================
X_tr, X_te, y_tr, y_te = train_test_split(X_imputed, y, test_size=0.2, random_state=RANDOM_STATE)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_tr, y_tr)

y_pred = rf.predict(X_te)
r2 = r2_score(y_te, y_pred)
rmse = mean_squared_error(y_te, y_pred, squared=False)
print(f"Model test performance: R2={r2:.4f}, RMSE={rmse:.4f}")

pd.DataFrame({"Metric": ["R2_test", "RMSE_test"], "Value": [r2, rmse]}).to_csv(
    os.path.join(OUT_DIR, "model_metrics.csv"),
    index=False, float_format="%.6g", encoding="utf-8-sig"
)

# =========================
# Step 3: SHAP analysis
# =========================
MAX_SAMPLES_FOR_SHAP = 5000
if len(X_imputed) > MAX_SAMPLES_FOR_SHAP:
    X_shap = X_imputed.sample(MAX_SAMPLES_FOR_SHAP, random_state=RANDOM_STATE)
else:
    X_shap = X_imputed

explainer = shap.TreeExplainer(rf)
shap_values_raw = explainer.shap_values(X_shap)
shap_values = getattr(shap_values_raw, "values", shap_values_raw)
if shap_values is None:
    raise RuntimeError("Failed to compute SHAP values.")

shap_df = pd.DataFrame(shap_values, columns=X_shap.columns, index=X_shap.index)
shap_df.to_csv(os.path.join(OUT_DIR, "shap_values.csv"), index=True, float_format="%.6g", encoding="utf-8-sig")
print("Saved shap_values.csv")

importance = np.abs(shap_values).mean(axis=0)
importance_df = (
    pd.DataFrame({"Feature": X_shap.columns, "MeanAbsShap": importance})
    .sort_values("MeanAbsShap", ascending=False)
    .reset_index(drop=True)
)
importance_df.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"),
                     index=False, float_format="%.6g", encoding="utf-8-sig")
importance_df.head(10).to_csv(os.path.join(OUT_DIR, "top10_importance.csv"),
                              index=False, float_format="%.6g", encoding="utf-8-sig")
print("Saved shap_importance.csv and top10_importance.csv")

TOP_N = min(30, len(X_shap.columns))
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=TOP_N)
plt.title("SHAP Feature Importance", fontsize=22, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary_bar.png"), bbox_inches="tight", dpi=300)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_shap, show=False, max_display=TOP_N)
plt.title("SHAP Summary Plot", fontsize=22, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary_dot.png"), bbox_inches="tight", dpi=300)
plt.close()
print("Saved SHAP summary plots")

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))[:100]

top_indices = np.argsort(importance)[::-1][:10]
top_features = X_shap.columns[top_indices]
print("Top SHAP features:", top_features.tolist())

for feat in top_features:
    plt.figure()
    shap.dependence_plot(feat, shap_values, X_shap, show=False)
    plt.title(f"SHAP Dependence: {feat}", fontsize=22, weight="bold")
    plt.tight_layout()
    fname = f"dependence_{_sanitize_filename(feat)}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), bbox_inches='tight', dpi=300)
    plt.close()
print("Saved SHAP dependence plots")

# =========================
# Step 4: Genetic Algorithm
# =========================
def _infer_bounds_from_data(X_df):
    bounds = {}
    for col in X_df.columns:
        c = pd.to_numeric(X_df[col], errors="coerce")
        lo, hi = float(np.nanmin(c)), float(np.nanmax(c))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = 0.0, 1.0
        bounds[col] = (lo, hi)
    return bounds

def _clip_to_bounds(df, bounds):
    out = df.copy()
    for feat, (lo, hi) in bounds.items():
        if feat in out.columns:
            out[feat] = np.clip(out[feat], lo, hi)
    return out

def _apply_fixed_params(df, fixed_params):
    out = df.copy()
    for k, v in fixed_params.items():
        if k in out.columns:
            out[k] = float(v)
    return out

def _quantize_by_steps(df, feature_steps):
    if not feature_steps:
        return df
    out = df.copy()
    for feat, step in feature_steps.items():
        if feat in out.columns and step and step > 0:
            out[feat] = np.round(out[feat] / step) * step
    return out

def _apply_constraints_df(df, constraints):
    if not constraints:
        return df
    mask = np.ones(len(df), dtype=bool)
    for c in constraints:
        cur = c(df) if callable(c) else np.array(c, dtype=bool)
        mask &= cur
    return df.loc[mask].reset_index(drop=True)

if os.path.exists(FIXED_PARAMS_FILE):
    fp_df = pd.read_csv(FIXED_PARAMS_FILE)
    fixed_params = {str(r["Feature"]): float(r["Value"]) for _, r in fp_df.iterrows()}
    print(f"Loaded fixed parameters: {fixed_params}")
else:
    fixed_params = {}
    print("No fixed_params.csv found, all features will be tunable.")

bounds_override = {}
feature_steps = {}
constraints = []

GA_POP = 200
GA_GENS = 60
TOURN_SIZE = 3
CX_RATE = 0.9
MUT_RATE = 0.3
MUT_SCALE = 0.15
ELITE_K = 5
RNG = np.random.RandomState(RANDOM_STATE)

all_bounds = _infer_bounds_from_data(X_imputed)
feat_cols = list(X_imputed.columns)
tunable_feats = [c for c in feat_cols if c not in fixed_params]
if len(tunable_feats) == 0:
    raise ValueError("No tunable features left after fixing parameters.")

def _random_individual():
    vals = []
    for f in feat_cols:
        lo, hi = all_bounds[f]
        v = fixed_params[f] if f in fixed_params else RNG.uniform(lo, hi)
        vals.append(v)
    return np.array(vals, dtype=float)

def _evaluate_population(pop):
    df_pop = pd.DataFrame(pop, columns=feat_cols)
    df_pop = _apply_fixed_params(df_pop, fixed_params)
    df_pop = _quantize_by_steps(df_pop, feature_steps)
    df_pop = _clip_to_bounds(df_pop, all_bounds)
    df_pop = _apply_constraints_df(df_pop, constraints)
    if len(df_pop) == 0:
        return np.array([])
    preds = rf.predict(df_pop)
    return preds

def _tournament_select(pop, fitness, k):
    sel = []
    n = len(pop)
    for _ in range(k):
        idx = RNG.choice(n, size=TOURN_SIZE, replace=False)
        best = idx[np.argmax(fitness[idx])]
        sel.append(pop[best].copy())
    return np.array(sel)

def _uniform_crossover(p1, p2):
    mask = RNG.rand(len(p1)) < 0.5
    c1, c2 = p1.copy(), p2.copy()
    c1[mask], c2[mask] = p2[mask], p1[mask]
    return c1, c2

def _mutate(ind):
    for f in tunable_feats:
        if RNG.rand() < 0.5:
            lo, hi = all_bounds[f]
            width = (hi - lo) * MUT_SCALE
            ind[feat_cols.index(f)] += RNG.normal(0.0, width)
    return ind

def _repair_df(df):
    df = _apply_fixed_params(df, fixed_params)
    df = _quantize_by_steps(df, feature_steps)
    df = _clip_to_bounds(df, all_bounds)
    return _apply_constraints_df(df, constraints)

pop = np.array([_random_individual() for _ in range(GA_POP)])
fit = _evaluate_population(pop)
if fit.size == 0:
    raise ValueError("All initial candidates invalid, adjust constraints or bounds.")

history_rows = []

def _append_history(pop, fit):
    df_hist = pd.DataFrame(pop, columns=feat_cols)
    df_hist = _repair_df(df_hist)
    preds = rf.predict(df_hist)
    df_hist["pred_GI"] = preds
    df_hist = df_hist.sort_values("pred_GI", ascending=False).head(10)
    history_rows.append(df_hist)

_append_history(pop, fit)

for g in range(GA_GENS):
    elite_idx = np.argsort(fit)[-ELITE_K:]
    elite = pop[elite_idx].copy()
    mating_pool = _tournament_select(pop, fit, GA_POP - ELITE_K)

    offspring = []
    i = 0
    while i < len(mating_pool):
        p1 = mating_pool[i]
        p2 = mating_pool[i + 1] if i + 1 < len(mating_pool) else mating_pool[0]
        if RNG.rand() < CX_RATE:
            c1, c2 = _uniform_crossover(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()
        offspring.extend([c1, c2])
        i += 2
    offspring = np.array(offspring)[: GA_POP - ELITE_K]

    for j in range(len(offspring)):
        if RNG.rand() < MUT_RATE:
            offspring[j] = _mutate(offspring[j])

    pop = np.vstack([elite, offspring])
    df_pop = pd.DataFrame(pop, columns=feat_cols)
    df_pop = _repair_df(df_pop)
    if len(df_pop) == 0:
        df_pop = pd.DataFrame([_random_individual() for _ in range(GA_POP)], columns=feat_cols)
        df_pop = _repair_df(df_pop)

    pop = df_pop[feat_cols].to_numpy()
    fit = rf.predict(df_pop)
    _append_history(pop, fit)

    if (g + 1) % 10 == 0 or g == GA_GENS - 1:
        print(f"Generation {g+1}/{GA_GENS}, best GI = {fit.max():.6f}")

all_hist = pd.concat(history_rows, axis=0, ignore_index=True)
all_hist = all_hist.sort_values("pred_GI", ascending=False).drop_duplicates(subset=feat_cols).reset_index(drop=True)
ga_csv_path = os.path.join(OUT_DIR, "tuning_ga_candidates.csv")
all_hist.to_csv(ga_csv_path, index=False, float_format="%.6g", encoding="utf-8-sig")

best_row = all_hist.iloc[0].to_dict()
best_pred = float(all_hist.iloc[0]["pred_GI"])
print("Best predicted GI:", round(best_pred, 6))
print("Best parameter suggestion:")
for k in feat_cols:
    print(f"  {k}: {best_row[k]}")

GA_best_params = {k: best_row[k] for k in feat_cols}
GA_best_pred = best_pred
