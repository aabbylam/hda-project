# eq5d_round4_nestedcv_with_perfold_save.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# -------------------------
# Paths / setup
# -------------------------
name = 'eq5d_round3'
base_dir = '/rds/general/user/hsl121/home/hda_project/final_code/results'
results_dir = os.path.join(base_dir, name)
fig_dir = os.path.join(results_dir, 'figures')
models_dir = os.path.join(results_dir, 'models')
perfold_dir = os.path.join(results_dir, 'per_fold')
preds_dir = os.path.join(results_dir, 'per_fold_preds')

os.makedirs(fig_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(perfold_dir, exist_ok=True)
os.makedirs(preds_dir, exist_ok=True)

# -------------------------
# Load & prepare data
# -------------------------
eq5d = pd.read_csv('../rq1/rq1_cleaned_no_ae.csv')
scores = pd.read_excel('../data/Scores 6 Jan 2025_Prescribed_Completed Baseline PROMs.xlsx')

# GAD7 wide
gad7 = scores[scores['promName'] == 'GAD7'][['SID', 'Round', 'total_score']]
gad7_wide = gad7.pivot_table(index='SID', columns='Round', values='total_score', aggfunc='first')
gad7_wide.columns = [f"GAD7_Round{r}" for r in gad7_wide.columns]
gad7_wide = gad7_wide.reset_index()
gad7 = pd.merge(eq5d, gad7_wide, on='SID', how='left')

# Insomnia wide
ins = scores[scores['promName'] == 'insomniaEfficacyMeasure'][['SID', 'Round', 'total_score']]
ins_wide = ins.pivot_table(index='SID', columns='Round', values='total_score', aggfunc='first')
ins_wide.columns = [f"insomniaEfficacyMeasure_Round{r}" for r in ins_wide.columns]
ins_wide = ins_wide.reset_index()
full = pd.merge(gad7, ins_wide, on='SID', how='left')

# Target = EQ5D_Round4 (6 months)
drop_cols = [
    'SID',
    # drop future PROM rounds from predictors
    'GAD7_Round2','GAD7_Round3','GAD7_Round4','GAD7_Round5','GAD7_Round6','GAD7_Round7',
    'GAD7_Round8','GAD7_Round9','GAD7_Round10','GAD7_Round11','GAD7_Round12','GAD7_Round13',
    'EQ5D_Round2','EQ5D_Round3','EQ5D_Round4','EQ5D_Round5','EQ5D_Round6',
    'insomniaEfficacyMeasure_Round2','insomniaEfficacyMeasure_Round3','insomniaEfficacyMeasure_Round4',
    'insomniaEfficacyMeasure_Round5','insomniaEfficacyMeasure_Round6','insomniaEfficacyMeasure_Round7',
    'insomniaEfficacyMeasure_Round8','insomniaEfficacyMeasure_Round9','insomniaEfficacyMeasure_Round10',
    'insomniaEfficacyMeasure_Round11','insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13',
    'GAD7_Round1_y','insomniaEfficacyMeasure_Round1_y'
]

X = full.drop(columns=drop_cols, errors='ignore')
y = full['EQ5D_Round3']

# clean rows
data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='EQ5D_Round3')
y = data['EQ5D_Round3']

# fix duplicate suffixes if present
X = X.rename(columns={
    'GAD7_Round1_x': 'GAD7_Round1',
    'insomniaEfficacyMeasure_Round1_x': 'insomniaEfficacyMeasure_Round1'
})

# keep aligned SIDs for prediction exports
sid_aligned = full.loc[data.index, 'SID'] if 'SID' in full.columns else pd.Series(data.index, index=data.index)

# -------------------------
# Models & grids
# -------------------------
def get_models_and_grids():
    models = {
        'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso())]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))]),
        'XGB': Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(objective='reg:squarederror', random_state=42))]),
        'MLP': Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(random_state=42, max_iter=1000))])
    }
    grids = {
        'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
        'Lasso': {'model__alpha': [0.01, 0.1, 1.0]},
        'RandomForest': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2']  # avoid deprecated 'auto'
        },
        'XGB': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 6, 9, 12],
            'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0]
        },
        'MLP': {
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50), (300, 200, 100), (100, 100, 50, 25)],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam', 'sgd'],
            'model__alpha': [1e-5, 1e-4, 1e-3],
            'model__learning_rate_init': [1e-3, 1e-2],
            'model__batch_size': [32, 64, 128, 256]
        }
    }
    return models, grids

models, grids = get_models_and_grids()

# -------------------------
# Nested CV with shared outer folds + per-fold logging
# -------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
outer_splits = list(outer_cv.split(X, y))  # share across models
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)

ensemble_results = {}
fold_r2 = {m: [] for m in models}
fold_mse = {m: [] for m in models}
fold_mae = {m: [] for m in models}
per_fold_rows = []

for model_name, pipe in models.items():
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grids[model_name],
        cv=inner_cv,
        scoring='neg_mean_absolute_error',  # MAE for tuning as you had
        n_jobs=-1,
        refit=True,
        return_train_score=False
    )

    for fold_id, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        gs.fit(X_tr, y_tr)
        preds = gs.predict(X_te)

        r2  = r2_score(y_te, preds)
        mse = mean_squared_error(y_te, preds)
        mae = mean_absolute_error(y_te, preds)

        fold_r2[model_name].append(r2)
        fold_mse[model_name].append(mse)
        fold_mae[model_name].append(mae)

        # log row
        per_fold_rows.append({
            "analysis": name,
            "model": model_name,
            "fold": fold_id,
            "r2": r2,
            "mse": mse,
            "mae": mae,
            "best_params": gs.best_params_
        })

        # save predictions for diagnostics
        sid_fold = sid_aligned.iloc[te_idx].reset_index(drop=True)
        pred_df = pd.DataFrame({
            "SID": sid_fold,
            "y_true": y_te.values,
            "y_pred": preds
        })
        pred_df.to_csv(os.path.join(preds_dir, f"{name}_{model_name}_fold{fold_id}_preds.csv"), index=False)

    # aggregate per-model statistics
    r2_arr, mse_arr, mae_arr = map(np.asarray, [fold_r2[model_name], fold_mse[model_name], fold_mae[model_name]])
    ensemble_results[model_name] = {
        'best_params': gs.best_params_,
        'r2_mean':  r2_arr.mean(),
        'r2_std':   r2_arr.std(ddof=1),
        'mse_mean': mse_arr.mean(),
        'mse_std':  mse_arr.std(ddof=1),
        'mse_se':   mse_arr.std(ddof=1) / np.sqrt(len(mse_arr)),
        'mae_mean': mae_arr.mean(),
        'mae_std':  mae_arr.std(ddof=1),
        'mae_se':   mae_arr.std(ddof=1) / np.sqrt(len(mae_arr))
    }

# save per-fold metrics (enables paired tests later)
per_fold_df = pd.DataFrame(per_fold_rows)
per_fold_df.to_csv(os.path.join(perfold_dir, f"{name}_per_fold_metrics.csv"), index=False)

# save summary results
results_df = pd.DataFrame.from_dict(ensemble_results, orient='index')
results_df.index.name = 'Model'
results_df.reset_index(inplace=True)
results_df.to_csv(os.path.join(results_dir, f'{name}_results.csv'), index=False)

# -------------------------
# Refit on full data & save pipelines
# -------------------------
for model_name, pipe in models.items():
    best_params = ensemble_results[model_name]['best_params']
    # convert GridSearch key names back to pipeline param style
    final_params = {k: v for k, v in best_params.items()}
    final_pipe = pipe.set_params(**final_params)
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

# -------------------------
# Plots
# -------------------------
# 1) MAE & MSE bar with SE
plt.figure(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35
colors = ['#4e79a7', '#f28e2b']  # MAE blue, MSE orange

plt.bar(x - width/2, results_df['mae_mean'], width, yerr=results_df['mae_se'], capsize=5,
        color=colors[0], label='MAE')
plt.bar(x + width/2, results_df['mse_mean'], width, yerr=results_df['mse_se'], capsize=5,
        color=colors[1], label='MSE')

plt.xticks(x, results_df['Model'], rotation=45, ha='right')
plt.ylabel('Error Value')
plt.title(f'{name}: Model Performance Comparison (5-fold CV)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_error_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2) Distributions across folds (R2, MSE, MAE)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
boxprops = dict(linewidth=1.5, facecolor='white')
medianprops = dict(color='black', linewidth=2)
whiskerprops = dict(linewidth=1.5)
capprops = dict(linewidth=1.5)
metric_colors = ['#59a14f', '#e15759', '#76b7b2']  # green, red, teal

axes[0].boxplot([fold_r2[m] for m in models.keys()], labels=models.keys(),
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)
for box in axes[0].artists:
    box.set_facecolor(metric_colors[0])
axes[0].set_ylabel('R²')
axes[0].set_title('R² Distribution Across Folds')

axes[1].boxplot([fold_mse[m] for m in models.keys()], labels=models.keys(),
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)
for box in axes[1].artists:
    box.set_facecolor(metric_colors[1])
axes[1].set_ylabel('MSE')
axes[1].set_title('MSE Distribution Across Folds')

axes[2].boxplot([fold_mae[m] for m in models.keys()], labels=models.keys(),
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops, patch_artist=True)
for box in axes[2].artists:
    box.set_facecolor(metric_colors[2])
axes[2].set_ylabel('MAE')
axes[2].set_title('MAE Distribution Across Folds')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_cv_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3) MSE and R² comparison bars with SD
plt.figure(figsize=(10, 6))
mse_means = [np.mean(fold_mse[m]) for m in models.keys()]
mse_stds  = [np.std(fold_mse[m], ddof=1) for m in models.keys()]
model_names = list(models.keys())
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
bars = plt.bar(model_names, mse_means, yerr=mse_stds, capsize=10, color=colors, width=0.6)
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Cross-Validated MSE by Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=10)
plt.ylim(min(mse_means)*0.98, max(mse_means)*1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_mse_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
r2_means = [np.mean(fold_r2[m]) for m in models.keys()]
r2_stds  = [np.std(fold_r2[m], ddof=1) for m in models.keys()]
bars = plt.bar(model_names, r2_means, yerr=r2_stds, capsize=10, color=colors, width=0.6)
plt.ylabel('R² (Mean)')
plt.title('Cross-Validated R² by Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=10)
plt.ylim(min(r2_means)*0.98, max(r2_means)*1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_r2_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
