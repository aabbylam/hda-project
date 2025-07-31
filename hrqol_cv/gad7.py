import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

name = 'gad7_round2'
base_dir = '/rds/general/user/hsl121/home/hda_project/hrqol_cv/results'
results_dir = os.path.join(base_dir, name)
fig_dir = os.path.join(results_dir, 'figures')
models_dir = os.path.join(results_dir, 'models')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Load data
eq5d = pd.read_csv('../rq1/rq1_cleaned_no_ae.csv')
scores = pd.read_excel('../data/Scores 6 Jan 2025_Prescribed_Completed Baseline PROMs.xlsx')

gad7 = scores[scores['promName']=='GAD7'][['SID','Round','total_score']]
gad7_wide = gad7.pivot_table(index='SID', columns='Round', values='total_score', aggfunc='first')
gad7_wide.columns = [f"GAD7_Round{r}" for r in gad7_wide.columns]
gad7_wide = gad7_wide.reset_index()
gad7 = pd.merge(eq5d, gad7_wide, on='SID', how='left')

ins = scores[scores['promName']=='insomniaEfficacyMeasure'][['SID','Round','total_score']]
ins_wide = ins.pivot_table(index='SID', columns='Round', values='total_score', aggfunc='first')
ins_wide.columns = [f"insomniaEfficacyMeasure_Round{r}" for r in ins_wide.columns]
ins_wide = ins_wide.reset_index()
full = pd.merge(gad7, ins_wide, on='SID', how='left')

drop_cols = [
    'SID', 'GAD7_Round2','GAD7_Round3','GAD7_Round4','GAD7_Round5','GAD7_Round6','GAD7_Round7',
    'GAD7_Round8','GAD7_Round9','GAD7_Round10','GAD7_Round11','GAD7_Round12','GAD7_Round13',
    'EQ5D_Round2','EQ5D_Round3','EQ5D_Round4','EQ5D_Round5','EQ5D_Round6',
    'insomniaEfficacyMeasure_Round2','insomniaEfficacyMeasure_Round3','insomniaEfficacyMeasure_Round4',
    'insomniaEfficacyMeasure_Round5','insomniaEfficacyMeasure_Round6','insomniaEfficacyMeasure_Round7',
    'insomniaEfficacyMeasure_Round8','insomniaEfficacyMeasure_Round9','insomniaEfficacyMeasure_Round10',
    'insomniaEfficacyMeasure_Round11','insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13'
]

X = full.drop(columns=drop_cols)
y = full['GAD7_Round2']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='GAD7_Round2'), data['GAD7_Round2']

# Define models and grids
models = {
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=0))]),
    'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=0))]),
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
        'model__max_features': ['auto', 'sqrt', 'log2']
    },
    'XGB': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 6, 9, 12],
        'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0]
    },
    'MLP': {
        'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200,100),(200,100,50), (300,200,100), (100,100,50,25)],
        'model__activation': ['relu', 'tanh'],
        'model__solver': ['adam', 'sgd'],
        'model__alpha': [1e-5, 1e-4, 1e-3],
        'model__learning_rate_init': [1e-3, 1e-2],
        'model__batch_size': [32, 64, 128, 256]
    }
}

# Nested CV
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)

fold_results = []
fold_metrics = {m: {'R2': [], 'MSE': [], 'RMSE': [], 'MAE': []} for m in models}

for name, pipe in models.items():
    print(f"Running nested CV: {name}")
    gs = GridSearchCV(pipe, grids[name], cv=inner_cv, scoring='r2', n_jobs=-1)
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = gs.predict(X.iloc[test_idx])
        true = y.iloc[test_idx]

        r2 = r2_score(true, preds)
        mse = mean_squared_error(true, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, preds)

        fold_metrics[name]['R2'].append(r2)
        fold_metrics[name]['MSE'].append(mse)
        fold_metrics[name]['RMSE'].append(rmse)
        fold_metrics[name]['MAE'].append(mae)

        fold_results.append({'Model': name, 'Fold': fold+1, 'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae})

fold_df = pd.DataFrame(fold_results)
fold_df.to_csv(os.path.join(results_dir, f'{name}_cv_folds.csv'), index=False)

# Summary stats
summary = []
for name in models:
    summary.append({
        'Model': name,
        'R2_mean': np.mean(fold_metrics[name]['R2']),
        'R2_se': np.std(fold_metrics[name]['R2']) / np.sqrt(5),
        'MSE_mean': np.mean(fold_metrics[name]['MSE']),
        'MSE_se': np.std(fold_metrics[name]['MSE']) / np.sqrt(5),
        'RMSE_mean': np.mean(fold_metrics[name]['RMSE']),
        'RMSE_se': np.std(fold_metrics[name]['RMSE']) / np.sqrt(5),
        'MAE_mean': np.mean(fold_metrics[name]['MAE']),
        'MAE_se': np.std(fold_metrics[name]['MAE']) / np.sqrt(5),
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(results_dir, f'{name}_summary_metrics.csv'), index=False)

# Final model refit
best_model = summary_df.sort_values('MSE_mean').iloc[0]['Model']
print(f"\nBest model: {best_model}")
final_gs = GridSearchCV(models[best_model], grids[best_model], cv=inner_cv, scoring='r2', n_jobs=-1)
final_gs.fit(X, y)
joblib.dump(final_gs.best_estimator_, os.path.join(models_dir, f'{name}_{best_model}_final.pkl'))

# Boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
pd.DataFrame({k: v['R2'] for k, v in fold_metrics.items()}).plot.box(ax=axes[0])
axes[0].set_title("R² (CV)")
axes[0].set_ylabel("R²")

pd.DataFrame({k: v['MSE'] for k, v in fold_metrics.items()}).plot.box(ax=axes[1])
axes[1].set_title("MSE (CV)")
axes[1].set_ylabel("MSE")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_cv_boxplots.png'), dpi=300)
plt.close()
