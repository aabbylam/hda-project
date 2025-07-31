import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib



name = 'eq5d_round2'
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
y = full['EQ5D_Round2']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='EQ5D_Round2'), data['EQ5D_Round2']

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
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200,100),(200,100,50), (300,200,100), (100,100,50,25)],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam', 'sgd'],
            'model__alpha': [1e-5, 1e-4, 1e-3],
            'model__learning_rate_init': [1e-3, 1e-2],
            'model__batch_size': [32, 64, 128, 256]
        }
    }

    return models, grids

cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

models, grids = get_models_and_grids()
results = []

for model_name, model_pipeline in models.items():
    grid = GridSearchCV(model_pipeline, grids[model_name], cv=cv, scoring=mse_scorer, n_jobs=-1, refit=True)
    grid.fit(X, y)

    best_model = grid.best_estimator_

    # Cross-validated scores
    neg_mse_scores = cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_squared_error')
    mae_scores     = cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores      = cross_val_score(best_model, X, y, cv=cv, scoring='r2')

    # Convert negatives
    mse_scores = -neg_mse_scores
    rmse_scores = np.sqrt(mse_scores)
    mae_scores = -mae_scores

    results.append({
        'Model': model_name,
        'Best Params': grid.best_params_,
        'Mean MSE': round(np.mean(mse_scores), 5),
        'Std Error MSE': round(np.std(mse_scores) / np.sqrt(len(mse_scores)), 5),
        'Mean RMSE': round(np.mean(rmse_scores), 5),
        'Std Error RMSE': round(np.std(rmse_scores) / np.sqrt(len(rmse_scores)), 5),
        'Mean MAE': round(np.mean(mae_scores), 5),
        'Std Error MAE': round(np.std(mae_scores) / np.sqrt(len(mae_scores)), 5),
        'Mean R2': round(np.mean(r2_scores), 5),
        'Std Error R2': round(np.std(r2_scores) / np.sqrt(len(r2_scores)), 5),
    })

    # Save best model
    joblib.dump(best_model, os.path.join(models_dir, f'{name}_{model_name}.pkl'))


results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, f'{name}_gridsearch_cv_full_metrics.csv'), index=False)

# Boxplots
r2_box = {}
mse_box = {}

for model_name, model_pipeline in models.items():
    print(f"Collecting fold scores for {model_name}")
    grid = GridSearchCV(model_pipeline, grids[model_name], cv=cv, scoring=mse_scorer, n_jobs=-1, refit=True)
    grid.fit(X, y)
    best_model = grid.best_estimator_

    # Store per-fold scores
    r2_scores = cross_val_score(best_model, X, y, cv=cv, scoring='r2')
    mse_scores = -cross_val_score(best_model, X, y, cv=cv, scoring='neg_mean_squared_error')

    r2_box[model_name] = r2_scores
    mse_box[model_name] = mse_scores

# === Convert to DataFrames ===
r2_df = pd.DataFrame(r2_box)
mse_df = pd.DataFrame(mse_box)

# === Plot R² Boxplot ===
plt.figure(figsize=(10, 5))
sns.boxplot(data=r2_df)
plt.title('Cross-Validated R² by Model')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_r2_boxplot.png'), dpi=300)
plt.close()

# === Plot MSE Boxplot ===
plt.figure(figsize=(10, 5))
sns.boxplot(data=mse_df)
plt.title('Cross-Validated MSE by Model')
plt.ylabel('Mean Squared Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_mse_boxplot.png'), dpi=300)
plt.close()