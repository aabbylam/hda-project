import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from matplotlib.lines import Line2D

# --- Your existing setup code ---
name = 'eq5d_round2'
base_dir = '/rds/general/user/hsl121/home/hda_project/hrqol/results'
models_dir = os.path.join(base_dir, name, 'models')
fig_dir = os.path.join(base_dir, name, 'figures_shap')
os.makedirs(fig_dir, exist_ok=True)

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
    'insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13', 'GAD7_Round1_y', 'insomniaEfficacyMeasure_Round1_y'
]
X = full.drop(columns=drop_cols)
y = full['EQ5D_Round2']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='EQ5D_Round2'), data['EQ5D_Round2']

X = X.rename(columns={
    'GAD7_Round1_x': 'GAD7_Round1',
    'insomniaEfficacyMeasure_Round1_x': 'insomniaEfficacyMeasure_Round1'})

# --- New: Initialize storage for comparison ---
shap_results = {}
summary_table = pd.DataFrame()

# --- Your existing model loop (modified with fix) ---
for file in os.listdir(models_dir):
    if not file.endswith('.pkl') or name not in file:
        continue

    model_path = os.path.join(models_dir, file)
    model = joblib.load(model_path)
    model_name = file.replace(f"{name}_", "").replace(".pkl", "")
    print(f"Processing: {model_name}")

    # If pipeline, extract core model and transform data
    if hasattr(model, 'named_steps'):
        X_scaled = model.named_steps['scaler'].transform(X)
        core_model = model.named_steps['model']
    else:
        X_scaled = X.copy()
        core_model = model

    # Select SHAP explainer based on model type
    if 'RandomForest' in model_name or 'XGB' in model_name:
        explainer = shap.TreeExplainer(core_model)
        shap_values = explainer.shap_values(X_scaled)
    elif 'Ridge' in model_name or 'Lasso' in model_name:
        explainer = shap.LinearExplainer(core_model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
    else:  # fallback for MLP or unsupported models
        explainer = shap.KernelExplainer(core_model.predict, shap.sample(X_scaled, 100))
        shap_values = explainer.shap_values(X_scaled[:100])

    # Feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]

    # --- FIXED: Handle different SHAP value structures ---
    # Determine the correct SHAP values to use
    if isinstance(shap_values, list):
        # Multi-class classification - use class 1 (positive class)
        if len(shap_values) > 1:
            plot_shap_values = shap_values[1]
        else:
            plot_shap_values = shap_values[0]
    else:
        # Regression or binary classification
        plot_shap_values = shap_values

    # Ensure we have a 2D array for plotting
    if plot_shap_values.ndim == 1:
        # If 1D, reshape to 2D (single sample case)
        plot_shap_values = plot_shap_values.reshape(1, -1)
        X_plot = X_scaled[:1] if len(X_scaled.shape) == 2 else X_scaled.reshape(1, -1)
    else:
        X_plot = X_scaled

    print(f"SHAP values shape: {plot_shap_values.shape}")
    print(f"X_plot shape: {X_plot.shape}")

    # --- Your existing plots (with fixed SHAP values) ---
    # Summary bar plot
    try:
        shap.summary_plot(plot_shap_values, X_plot, feature_names=feature_names, plot_type='bar', show=False)
        plt.title(f"SHAP Feature Importance: {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'shap_bar_{model_name}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot for {model_name}: {e}")

    # Full summary dot plot
    try:
        shap.summary_plot(plot_shap_values, X_plot, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary: {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'shap_summary_{model_name}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot for {model_name}: {e}")

    # --- New: Store SHAP results for comparison ---
    # Calculate mean absolute SHAP values for comparison
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            mean_shap = np.abs(shap_values[1]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values[0]).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    shap_results[model_name] = pd.Series(mean_shap, index=feature_names)
    summary_table[model_name] = shap_results[model_name]

# --- New: Comparative Analysis Section ---
# 1. Save summary table to CSV
summary_table.to_csv(os.path.join(fig_dir, 'shap_comparison_table.csv'))
shap_results_df = pd.DataFrame(shap_results)
shap_results_df.to_csv(os.path.join(fig_dir, 'shap_results_summary.csv'))
