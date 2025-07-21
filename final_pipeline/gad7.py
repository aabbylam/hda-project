import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor


name = 'gad7_round2'
results_dir = '/rds/general/user/hsl121/home/hda_project/final_pipeline/results'
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

# Prepare features and target
drop_cols = [
    'SID', 'GAD7_Round2','GAD7_Round3','GAD7_Round4','GAD7_Round5','GAD7_Round6','GAD7_Round7',
    'GAD7_Round8','GAD7_Round9','GAD7_Round10','GAD7_Round11','GAD7_Round12',
    'GAD7_Round13', 'EQ5D_Round2','EQ5D_Round3','EQ5D_Round4','EQ5D_Round5',
    'EQ5D_Round6', 'insomniaEfficacyMeasure_Round2','insomniaEfficacyMeasure_Round3',
    'insomniaEfficacyMeasure_Round4','insomniaEfficacyMeasure_Round5',
    'insomniaEfficacyMeasure_Round6','insomniaEfficacyMeasure_Round7',
    'insomniaEfficacyMeasure_Round8','insomniaEfficacyMeasure_Round9',
    'insomniaEfficacyMeasure_Round10','insomniaEfficacyMeasure_Round11',
    'insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13'
]
X = full.drop(columns=drop_cols)
y = full['GAD7_Round2']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='GAD7_Round2'), data['GAD7_Round2']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models and parameter grids
def get_models_and_grids():
    models = {
        'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=0))]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=0))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))]),
        'XGB': Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(objective='reg:squarederror', random_state=42))]),
        'MLP': Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(random_state=42, max_iter=1000))])
    }
    grids = {
        'Ridge': {'model__alpha': [0.1,1.0,10.0]},
        'Lasso': {'model__alpha': [0.01,0.1,1.0]},
        'RandomForest': {
            'model__n_estimators': [100,200,500],
            'model__max_depth': [None,5,10,20],
            'model__min_samples_split': [2,5,10],
            'model__min_samples_leaf': [1,2,4],
            'model__max_features': ['auto','sqrt','log2']
        },
        'XGB': {
            'model__n_estimators': [100,200,300],
            'model__max_depth': [3,6,9,12],
            'model__learning_rate': [0.001,0.01,0.1,0.2],
            'model__subsample': [0.6,0.8,1.0],
            'model__colsample_bytree': [0.6,0.8,1.0]
        },
        'MLP': {
            
            'model__hidden_layer_sizes': [(100,50), (100,50,25),(200,100,50),(200,100,50,25)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [1e-5, 1e-4, 1e-3],
            'model__learning_rate_init': [1e-3, 1e-2]
        }
    }

    return models, grids

models, grids = get_models_and_grids()

# Nested CV to evaluate and select hyperparameters
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)
ensemble_results = {}
fold_r2 = {m: [] for m in models}
fold_mse = {m: [] for m in models}

for model_name, pipe in models.items():
    gs = GridSearchCV(pipe, grids[model_name], cv=inner_cv, scoring='r2', n_jobs=-1)
    for tr_idx, te_idx in outer_cv.split(X):
        gs.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = gs.predict(X.iloc[te_idx])
        fold_r2[model_name].append(r2_score(y.iloc[te_idx], preds))
        fold_mse[model_name].append(mean_squared_error(y.iloc[te_idx], preds))
    ensemble_results[model_name] = {
        'best_params': gs.best_params_,
        'r2_mean': np.mean(fold_r2[model_name]),
        'r2_std': np.std(fold_r2[model_name]),
        'mse_mean': np.mean(fold_mse[model_name]),
        'mse_std': np.std(fold_mse[model_name])
    }

# Save results to CSV
results_df = pd.DataFrame.from_dict(ensemble_results, orient='index')
results_df.index.name = 'Model'
results_df.reset_index(inplace=True)
results_df.to_csv(os.path.join(results_dir, f'{name}_results.csv'), index=False)

# Re-fit each model on full data and save pipeline
for model_name, pipe in models.items():
    best_params = ensemble_results[model_name]['best_params']
    final_pipe = pipe.set_params(**{f"model__{k.split('__')[-1]}": v for k, v in best_params.items()})
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

# 6a) Bar-chart of mean R² ± SD
fig, ax = plt.subplots(figsize=(6,4))
plot_df = results_df.sort_values('r2_mean')
ax.barh(plot_df['Model'], plot_df['r2_mean'], xerr=plot_df['r2_std'],
        align='center', ecolor='gray', capsize=4)
ax.set_xlabel('Mean R² (± SD)')
ax.set_title('Model Comparison')
plt.tight_layout()
fig.savefig(os.path.join(fig_dir, f'{name}_model_comparison_r2.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

# --- 6b) Box‐plots of R² and MSE distributions across folds ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Prepare DataFrames: fold_r2 and fold_mse are dicts of lists
r2_box = pd.DataFrame(fold_r2)
mse_box = pd.DataFrame(fold_mse)

# R² box‐plot
r2_box.plot.box(ax=axes[0])
axes[0].set_ylabel('R²')
axes[0].set_title('CV R² Distributions')

# MSE box‐plot
mse_box.plot.box(ax=axes[1])
axes[1].set_ylabel('MSE')
axes[1].set_title('CV MSE Distributions')

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, f'{name}_cv_distributions_r2_mse.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)