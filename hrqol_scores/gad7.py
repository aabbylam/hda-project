import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

name = 'gad7_round4'
base_dir = '/rds/general/user/hsl121/home/hda_project/hrqol/results'
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
    'insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13','GAD7_Round1_y', 'insomniaEfficacyMeasure_Round1_y']

X = full.drop(columns=drop_cols)
y = full['GAD7_Round4']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='GAD7_Round4'), data['GAD7_Round4']

X=X.rename(columns={
    'GAD7_Round1_x': 'GAD7_Round1',
    'insomniaEfficacyMeasure_Round1_x': 'insomniaEfficacyMeasure_Round1'})


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

models, grids = get_models_and_grids()

# Nested CV to evaluate and select hyperparameters
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)
ensemble_results = {}
fold_r2 = {m: [] for m in models}
fold_mse = {m: [] for m in models}
fold_mae = {m: [] for m in models}

for model_name, pipe in models.items():
    gs = GridSearchCV(pipe, grids[model_name], cv=inner_cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    for tr_idx, te_idx in outer_cv.split(X):
        gs.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = gs.predict(X.iloc[te_idx])
        fold_r2[model_name].append(r2_score(y.iloc[te_idx], preds))
        fold_mse[model_name].append(mean_squared_error(y.iloc[te_idx], preds))
        fold_mae[model_name].append(mean_absolute_error(y.iloc[te_idx], preds))  # Add MAE calculation
    
    ensemble_results[model_name] = {
        'best_params': gs.best_params_,
        'r2_mean': np.mean(fold_r2[model_name]),
        'r2_std': np.std(fold_r2[model_name]),
        'mse_mean': np.mean(fold_mse[model_name]),
        'mse_std': np.std(fold_mse[model_name]),
        'mse_se': np.std(fold_mse[model_name])/np.sqrt(len(fold_mse[model_name])),  # Standard error
        'mae_mean': np.mean(fold_mae[model_name]),
        'mae_std': np.std(fold_mae[model_name]),
        'mae_se': np.std(fold_mae[model_name])/np.sqrt(len(fold_mae[model_name]))   # Standard error
    }

# Save comprehensive results to CSV
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

# 1. Performance Comparison Plot (MAE and MSE with standard error)
plt.figure(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35 

# Use a professional color palette
colors = ['#4e79a7', '#f28e2b']  # Blue for MAE, orange for MSE

# MAE plot with standard error
plt.bar(x - width/2, results_df['mae_mean'], width, 
        yerr=results_df['mae_se'], capsize=5,
        color=colors[0], label='MAE')

# MSE plot with standard error
plt.bar(x + width/2, results_df['mse_mean'], width,
        yerr=results_df['mse_se'], capsize=5,
        color=colors[1], label='MSE')

plt.xticks(x, results_df['Model'], rotation=45, ha='right')
plt.ylabel('Error Value')
plt.title(f'{name}: Model Performance Comparison (5-fold CV)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_error_comparison.png'),
           dpi=300, bbox_inches='tight')
plt.close()

# 2. Updated Distribution Boxplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Three subplots for R², MSE, MAE

# Professional boxplot styling
boxprops = dict(linewidth=1.5, facecolor='white')
medianprops = dict(color='black', linewidth=2)
whiskerprops = dict(linewidth=1.5)
capprops = dict(linewidth=1.5)

# Define colors for each metric
metric_colors = ['#59a14f', '#e15759', '#76b7b2']  # Green, red, teal

# R² Distribution
axes[0].boxplot(
    [fold_r2[m] for m in models.keys()],
    labels=models.keys(),
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    patch_artist=True
)
# Set color for each model's box
for i, box in enumerate(axes[0].artists):
    box.set_facecolor(metric_colors[0])
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² Distribution Across Folds')

# MSE Distribution
axes[1].boxplot(
    [fold_mse[m] for m in models.keys()],
    labels=models.keys(),
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    patch_artist=True
)
for i, box in enumerate(axes[1].artists):
    box.set_facecolor(metric_colors[1])
axes[1].set_ylabel('MSE')
axes[1].set_title('MSE Distribution Across Folds')

# MAE Distribution (new)
axes[2].boxplot(
    [fold_mae[m] for m in models.keys()],
    labels=models.keys(),
    boxprops=boxprops,
    medianprops=medianprops,
    whiskerprops=whiskerprops,
    capprops=capprops,
    patch_artist=True
)
for i, box in enumerate(axes[2].artists):
    box.set_facecolor(metric_colors[2])
axes[2].set_ylabel('MAE')
axes[2].set_title('MAE Distribution Across Folds')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_cv_distributions.png'),
           dpi=300, bbox_inches='tight')
plt.close()


# After your nested CV code, add this visualization:

plt.figure(figsize=(10, 6))

# Prepare MSE data (convert from your fold_mse dictionary)
mse_means = [np.mean(fold_mse[model]) for model in models.keys()]
mse_stds = [np.std(fold_mse[model]) for model in models.keys()]
model_names = list(models.keys())

# Custom colors - use your preferred palette
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

# Create the bar plot with error bars
bars = plt.bar(model_names, mse_means, 
               yerr=mse_stds,
               capsize=10,
               color=colors,
               width=0.6)

# Custom styling
plt.ylabel('Mean Squared Error (MSE)', fontsize=12, labelpad=10)
plt.title('Cross-Validated MSE by Model', fontsize=14, pad=20)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10)

# Adjust y-axis limits based on your data range
plt.ylim(min(mse_means)*0.98, max(mse_means)*1.02)  # 2% padding

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_mse_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# R² Comparison Bar Plot
plt.figure(figsize=(10, 6))

# Prepare R² data
r2_means = [np.mean(fold_r2[model]) for model in models.keys()]
r2_stds = [np.std(fold_r2[model]) for model in models.keys()]
model_names = list(models.keys())

# Custom colors (reuse same palette for consistency)
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

# Create the bar plot with error bars
bars = plt.bar(model_names, r2_means,
               yerr=r2_stds,
               capsize=10,
               color=colors,
               width=0.6)

# Custom styling
plt.ylabel('R² Score (Mean)', fontsize=12, labelpad=10)
plt.title('Cross-Validated R² Score by Model', fontsize=14, pad=20)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10)

# Adjust y-axis limits
plt.ylim(min(r2_means)*0.98, max(r2_means)*1.02)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_r2_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib   

# from sklearn.model_selection import train_test_split, GridSearchCV, KFold
# from sklearn.linear_model import Ridge, Lasso
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPRegressor

# # Job creation and execution functions for HPC
# def save_individual_jobs(X, y, models, grids, job_dir):
#     """Save individual training jobs as pickle files for HPC submission"""
#     os.makedirs(job_dir, exist_ok=True)
    
#     outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
#     inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)
    
#     job_id = 0
#     job_files = []
    
#     for model_name, pipe in models.items():
#         for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
#             job_data = {
#                 'model_name': model_name,
#                 'pipe': pipe,
#                 'grid': grids[model_name],
#                 'X_train': X.iloc[tr_idx],
#                 'X_test': X.iloc[te_idx],
#                 'y_train': y.iloc[tr_idx],
#                 'y_test': y.iloc[te_idx],
#                 'inner_cv': inner_cv,
#                 'job_id': job_id,
#                 'fold_idx': fold_idx
#             }
            
#             job_file = os.path.join(job_dir, f'job_{job_id}_{model_name}_fold_{fold_idx}.pkl')
#             joblib.dump(job_data, job_file)
#             job_files.append(job_file)
#             job_id += 1
    
#     return job_files

# def run_single_job(job_file, results_dir):
#     """Run a single job - to be called by SLURM job"""
#     os.makedirs(results_dir, exist_ok=True)
#     job_data = joblib.load(job_file)
    
#     gs = GridSearchCV(
#         job_data['pipe'], 
#         job_data['grid'], 
#         cv=job_data['inner_cv'], 
#         scoring='neg_mean_absolute_error', 
#         n_jobs=-1  # Use all available cores on the node
#     )
    
#     gs.fit(job_data['X_train'], job_data['y_train'])
#     preds = gs.predict(job_data['X_test'])
    
#     result = {
#         'model_name': job_data['model_name'],
#         'fold_idx': job_data['fold_idx'],
#         'job_id': job_data['job_id'],
#         'best_params': gs.best_params_,
#         'r2': r2_score(job_data['y_test'], preds),
#         'mse': mean_squared_error(job_data['y_test'], preds),
#         'mae': mean_absolute_error(job_data['y_test'], preds)
#     }
    
#     result_file = os.path.join(results_dir, f'result_{job_data["job_id"]}_{job_data["model_name"]}_fold_{job_data["fold_idx"]}.pkl')
#     joblib.dump(result, result_file)
    
#     print(f"Completed job {job_data['job_id']}: {job_data['model_name']} fold {job_data['fold_idx']}")
#     return result

# def collect_slurm_results(results_dir, models):
#     """Collect results from SLURM jobs and organize them exactly as original code"""
#     result_files = [f for f in os.listdir(results_dir) if f.startswith('result_')]
    
#     ensemble_results = {}
#     fold_r2 = {m: [] for m in models}
#     fold_mse = {m: [] for m in models}
#     fold_mae = {m: [] for m in models}
    
#     # Collect all results first
#     all_results = []
#     for result_file in result_files:
#         result = joblib.load(os.path.join(results_dir, result_file))
#         all_results.append(result)
    
#     # Sort results by model and fold to ensure consistent ordering
#     all_results.sort(key=lambda x: (x['model_name'], x['fold_idx']))
    
#     # Group by model and collect fold results
#     for result in all_results:
#         model_name = result['model_name']
#         fold_r2[model_name].append(result['r2'])
#         fold_mse[model_name].append(result['mse'])
#         fold_mae[model_name].append(result['mae'])
        
#         # Store best params (use the last one for consistency with original code)
#         ensemble_results[model_name] = {'best_params': result['best_params']}
    
#     # Calculate statistics exactly as in original code
#     for model_name in models:
#         if fold_r2[model_name]:  # Check if we have results
#             ensemble_results[model_name].update({
#                 'r2_mean': np.mean(fold_r2[model_name]),
#                 'r2_std': np.std(fold_r2[model_name]),
#                 'mse_mean': np.mean(fold_mse[model_name]),
#                 'mse_std': np.std(fold_mse[model_name]),
#                 'mse_se': np.std(fold_mse[model_name])/np.sqrt(len(fold_mse[model_name])),  # Standard error
#                 'mae_mean': np.mean(fold_mae[model_name]),
#                 'mae_std': np.std(fold_mae[model_name]),
#                 'mae_se': np.std(fold_mae[model_name])/np.sqrt(len(fold_mae[model_name]))   # Standard error
#             })
    
#     return ensemble_results, fold_r2, fold_mse, fold_mae

# # Main script - exactly as your original with HPC parallelization
# name = 'eq5d_round4'
# base_dir = '/rds/general/user/hsl121/home/hda_project/hrqol/results'
# results_dir = os.path.join(base_dir, name)
# fig_dir = os.path.join(results_dir, 'figures')
# models_dir = os.path.join(results_dir, 'models')
# jobs_dir = os.path.join(results_dir, 'jobs')
# hpc_results_dir = os.path.join(results_dir, 'hpc_results')

# os.makedirs(fig_dir, exist_ok=True)
# os.makedirs(models_dir, exist_ok=True)
# os.makedirs(jobs_dir, exist_ok=True)
# os.makedirs(hpc_results_dir, exist_ok=True)

# # Load data - exactly as your original
# eq5d = pd.read_csv('../rq1/rq1_cleaned_no_ae.csv')
# scores = pd.read_excel('../data/Scores 6 Jan 2025_Prescribed_Completed Baseline PROMs.xlsx')

# gad7 = scores[scores['promName']=='GAD7'][['SID','Round','total_score']]
# gad7_wide = gad7.pivot_table(index='SID', columns='Round', values='total_score', aggfunc='first')
# gad7_wide.columns = [f"GAD7_Round{r}" for r in gad7_wide.columns]
# gad7_wide = gad7_wide.reset_index()
# gad7 = pd.merge(eq5d, gad7_wide, on='SID', how='left')

# ins = scores[scores['promName']=='insomniaEfficacyMeasure'][['SID','Round','total_score']]
# ins_wide = ins.pivot_table(index='SID', columns='Round', values='total_score', aggfunc='first')
# ins_wide.columns = [f"insomniaEfficacyMeasure_Round{r}" for r in ins_wide.columns]
# ins_wide = ins_wide.reset_index()
# full = pd.merge(gad7, ins_wide, on='SID', how='left')

# # Prepare features and target - exactly as your original
# drop_cols = [
#     'SID', 'GAD7_Round2','GAD7_Round3','GAD7_Round4','GAD7_Round5','GAD7_Round6','GAD7_Round7',
#     'GAD7_Round8','GAD7_Round9','GAD7_Round10','GAD7_Round11','GAD7_Round12',
#     'GAD7_Round13', 'EQ5D_Round2','EQ5D_Round3','EQ5D_Round4','EQ5D_Round5',
#     'EQ5D_Round6', 'insomniaEfficacyMeasure_Round2','insomniaEfficacyMeasure_Round3',
#     'insomniaEfficacyMeasure_Round4','insomniaEfficacyMeasure_Round5',
#     'insomniaEfficacyMeasure_Round6','insomniaEfficacyMeasure_Round7',
#     'insomniaEfficacyMeasure_Round8','insomniaEfficacyMeasure_Round9',
#     'insomniaEfficacyMeasure_Round10','insomniaEfficacyMeasure_Round11',
#     'insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13'
# ]
# X = full.drop(columns=drop_cols)
# y = full['EQ5D_Round4']
# data = pd.concat([X, y], axis=1).dropna()
# X, y = data.drop(columns='EQ5D_Round4'), data['EQ5D_Round4']

# X=X.rename(columns={
#     'GAD7_Round1_x': 'GAD7_Round1',
#     'insomniaEfficacyMeasure_Round1_x': 'insomniaEfficacyMeasure_Round1'})

# # Define models and parameter grids - exactly as your original
# def get_models_and_grids():
#     models = {
#         'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=0))]),
#         'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=0))]),
#         'RandomForest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestRegressor(random_state=42))]),
#         'XGB': Pipeline([('scaler', StandardScaler()), ('model', XGBRegressor(objective='reg:squarederror', random_state=42))]),
#         'MLP': Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(random_state=42, max_iter=1000))])
#     }
#     grids = {
#         'Ridge': {'model__alpha': [0.1,1.0,10.0]},
#         'Lasso': {'model__alpha': [0.01,0.1,1.0]},
#         'RandomForest': {
#             'model__n_estimators': [100,200,500],
#             'model__max_depth': [None,5,10,20],
#             'model__min_samples_split': [2,5,10],
#             'model__min_samples_leaf': [1,2,4],
#             'model__max_features': ['auto','sqrt','log2']
#         },
#         'XGB': {
#             'model__n_estimators': [100,200,300],
#             'model__max_depth': [3,6,9,12],
#             'model__learning_rate': [0.001,0.01,0.1,0.2],
#             'model__subsample': [0.6,0.8,1.0],
#             'model__colsample_bytree': [0.6,0.8,1.0]
#         },
#         'MLP': {
#             'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200,100),(200,100,50), (300,200,100), (100,100,50,25)],
#             'model__activation': ['relu', 'tanh'],
#             'model__solver': ['adam', 'sgd'],
#             'model__alpha': [1e-5, 1e-4, 1e-3],
#             'model__learning_rate_init': [1e-3, 1e-2],
#             'model__batch_size': [32, 64, 128, 256]
#         }
#     }
#     return models, grids

# models, grids = get_models_and_grids()

# # HPC Execution Logic
# import sys
# if len(sys.argv) > 1:
#     # This is being called as a SLURM array job
#     if sys.argv[1] == 'run_job':
#         job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', sys.argv[2]))
#         job_files = [f for f in os.listdir(jobs_dir) if f.startswith('job_')]
#         job_files.sort()  # Ensure consistent ordering
        
#         if job_id < len(job_files):
#             job_file = os.path.join(jobs_dir, job_files[job_id])
#             run_single_job(job_file, hpc_results_dir)
#         else:
#             print(f"Job ID {job_id} is out of range")
        
#         sys.exit(0)
    
#     elif sys.argv[1] == 'collect_results':
#         # Collect results after all jobs complete
#         print("Collecting results from HPC jobs...")
#         ensemble_results, fold_r2, fold_mse, fold_mae = collect_slurm_results(hpc_results_dir, models)
        
#         # Continue with the rest of the original script...
        
# else:
#     # Initial setup - create job files
#     print("Creating HPC job files...")
#     job_files = save_individual_jobs(X, y, models, grids, jobs_dir)
#     print(f"Created {len(job_files)} job files in {jobs_dir}")
#     print(f"Total jobs: 25 (5 models × 5 folds)")
    
#     # Save job list for reference
#     joblib.dump(job_files, os.path.join(results_dir, 'job_list.pkl'))
    
#     print("\nTo run on HPC:")
#     print("1. Submit array job: sbatch --array=0-24 your_slurm_script.sh")
#     print("2. After completion: python this_script.py collect_results")
    
#     sys.exit(0)

# # Results processing and visualization (runs when collecting results)
# if 'ensemble_results' in locals():
    
#     # Save comprehensive results to CSV - exactly as original
#     results_df = pd.DataFrame.from_dict(ensemble_results, orient='index')
#     results_df.index.name = 'Model'
#     results_df.reset_index(inplace=True)
#     results_df.to_csv(os.path.join(results_dir, f'{name}_results.csv'), index=False)

#     # Re-fit each model on full data and save pipeline - exactly as original
#     for model_name, pipe in models.items():
#         best_params = ensemble_results[model_name]['best_params']
#         final_pipe = pipe.set_params(**{f"model__{k.split('__')[-1]}": v for k, v in best_params.items()})
#         final_pipe.fit(X, y)
#         joblib.dump(final_pipe, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

#     # 1. Performance Comparison Plot (MAE and MSE with standard error) - exactly as original
#     plt.figure(figsize=(10, 5))
#     x = np.arange(len(results_df))
#     width = 0.35 

#     # Use a professional color palette
#     colors = ['#4e79a7', '#f28e2b']  # Blue for MAE, orange for MSE

#     # MAE plot with standard error
#     plt.bar(x - width/2, results_df['mae_mean'], width, 
#             yerr=results_df['mae_se'], capsize=5,
#             color=colors[0], label='MAE')

#     # MSE plot with standard error
#     plt.bar(x + width/2, results_df['mse_mean'], width,
#             yerr=results_df['mse_se'], capsize=5,
#             color=colors[1], label='MSE')

#     plt.xticks(x, results_df['Model'], rotation=45, ha='right')
#     plt.ylabel('Error Value')
#     plt.title(f'{name}: Model Performance Comparison (5-fold CV)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, f'{name}_error_comparison.png'),
#                dpi=300, bbox_inches='tight')
#     plt.close()

#     # 2. Updated Distribution Boxplots - exactly as original
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Three subplots for R², MSE, MAE

#     # Professional boxplot styling
#     boxprops = dict(linewidth=1.5, facecolor='white')
#     medianprops = dict(color='black', linewidth=2)
#     whiskerprops = dict(linewidth=1.5)
#     capprops = dict(linewidth=1.5)

#     # Define colors for each metric
#     metric_colors = ['#59a14f', '#e15759', '#76b7b2']  # Green, red, teal

#     # R² Distribution
#     axes[0].boxplot(
#         [fold_r2[m] for m in models.keys()],
#         labels=models.keys(),
#         boxprops=boxprops,
#         medianprops=medianprops,
#         whiskerprops=whiskerprops,
#         capprops=capprops,
#         patch_artist=True
#     )
#     # Set color for each model's box
#     for i, box in enumerate(axes[0].artists):
#         box.set_facecolor(metric_colors[0])
#     axes[0].set_ylabel('R² Score')
#     axes[0].set_title('R² Distribution Across Folds')

#     # MSE Distribution
#     axes[1].boxplot(
#         [fold_mse[m] for m in models.keys()],
#         labels=models.keys(),
#         boxprops=boxprops,
#         medianprops=medianprops,
#         whiskerprops=whiskerprops,
#         capprops=capprops,
#         patch_artist=True
#     )
#     for i, box in enumerate(axes[1].artists):
#         box.set_facecolor(metric_colors[1])
#     axes[1].set_ylabel('MSE')
#     axes[1].set_title('MSE Distribution Across Folds')

#     # MAE Distribution (new)
#     axes[2].boxplot(
#         [fold_mae[m] for m in models.keys()],
#         labels=models.keys(),
#         boxprops=boxprops,
#         medianprops=medianprops,
#         whiskerprops=whiskerprops,
#         capprops=capprops,
#         patch_artist=True
#     )
#     for i, box in enumerate(axes[2].artists):
#         box.set_facecolor(metric_colors[2])
#     axes[2].set_ylabel('MAE')
#     axes[2].set_title('MAE Distribution Across Folds')

#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, f'{name}_cv_distributions.png'),
#                dpi=300, bbox_inches='tight')
#     plt.close()

#     # MSE Comparison Bar Plot - exactly as original
#     plt.figure(figsize=(10, 6))

#     # Prepare MSE data (convert from your fold_mse dictionary)
#     mse_means = [np.mean(fold_mse[model]) for model in models.keys()]
#     mse_stds = [np.std(fold_mse[model]) for model in models.keys()]
#     model_names = list(models.keys())

#     # Custom colors - use your preferred palette
#     colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

#     # Create the bar plot with error bars
#     bars = plt.bar(model_names, mse_means, 
#                    yerr=mse_stds,
#                    capsize=10,
#                    color=colors,
#                    width=0.6)

#     # Custom styling
#     plt.ylabel('Mean Squared Error (MSE)', fontsize=12, labelpad=10)
#     plt.title('Cross-Validated MSE by Model', fontsize=14, pad=20)
#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)

#     # Add value labels on top of bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                  f'{height:.4f}',
#                  ha='center', va='bottom', fontsize=10)

#     # Adjust y-axis limits based on your data range
#     plt.ylim(min(mse_means)*0.98, max(mse_means)*1.02)  # 2% padding

#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, f'{name}_mse_comparison.png'), 
#                 dpi=300, bbox_inches='tight')
#     plt.close()

#     # R² Comparison Bar Plot - exactly as original
#     plt.figure(figsize=(10, 6))

#     # Prepare R² data
#     r2_means = [np.mean(fold_r2[model]) for model in models.keys()]
#     r2_stds = [np.std(fold_r2[model]) for model in models.keys()]
#     model_names = list(models.keys())

#     # Custom colors (reuse same palette for consistency)
#     colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

#     # Create the bar plot with error bars
#     bars = plt.bar(model_names, r2_means,
#                    yerr=r2_stds,
#                    capsize=10,
#                    color=colors,
#                    width=0.6)

#     # Custom styling
#     plt.ylabel('R² Score (Mean)', fontsize=12, labelpad=10)
#     plt.title('Cross-Validated R² Score by Model', fontsize=14, pad=20)
#     plt.xticks(fontsize=11)
#     plt.yticks(fontsize=11)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)

#     # Add value labels on top of bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                  f'{height:.4f}',
#                  ha='center', va='bottom', fontsize=10)

#     # Adjust y-axis limits
#     plt.ylim(min(r2_means)*0.98, max(r2_means)*1.02)

#     plt.tight_layout()
#     plt.savefig(os.path.join(fig_dir, f'{name}_r2_comparison.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()

#     print("HPC nested CV completed successfully!")
#     print(f"Results saved to: {results_dir}")
#     print(f"Figures saved to: {fig_dir}")
#     print(f"Models saved to: {models_dir}")