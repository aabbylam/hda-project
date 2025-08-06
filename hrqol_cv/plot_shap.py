import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from matplotlib.lines import Line2D

# --- Your existing setup code ---
name = 'eq5d_round2'
base_dir = '/rds/general/user/hsl121/home/hda_project/hrqol_cv/results'
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

X=X.rename(columns={
    'GAD7_Round1_x': 'GAD7_Round1',
    'insomniaEfficacyMeasure_Round1_x': 'insomniaEfficacyMeasure_Round1'})


# --- New: Initialize storage for comparison ---
shap_results = {}
summary_table = pd.DataFrame()

# --- Your existing model loop (modified) ---
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
        class_idx = 1 if isinstance(shap_values, list) else 0
    elif 'Ridge' in model_name or 'Lasso' in model_name:
        explainer = shap.LinearExplainer(core_model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)
        class_idx = 1 if shap_values.ndim == 3 else 0
    else:  # fallback for MLP or unsupported models
        explainer = shap.KernelExplainer(core_model.predict_proba, shap.sample(X_scaled, 100))
        shap_values = explainer.shap_values(X_scaled[:100])
        class_idx = 1

    # Feature names
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]

    # --- Your existing plots ---
    # Summary bar plot
    shap.summary_plot(shap_values[class_idx], X_scaled, feature_names=feature_names, plot_type='bar', show=False)
    plt.title(f"SHAP Feature Importance: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'shap_bar_{model_name}.png'), dpi=300)
    plt.close()

    # Full summary dot plot
    shap.summary_plot(shap_values[class_idx], X_scaled, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary: {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'shap_summary_{model_name}.png'), dpi=300)
    plt.close()

    # --- New: Store SHAP results for comparison ---
    if isinstance(shap_values, list):
        mean_shap = np.abs(shap_values[class_idx]).mean(axis=0)
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    shap_results[model_name] = pd.Series(mean_shap, index=feature_names)
    summary_table[model_name] = shap_results[model_name]

# --- New: Comparative Analysis Section ---
# 1. Save summary table to CSV
summary_table.to_csv(os.path.join(fig_dir, 'shap_comparison_table.csv'))

# 2. Create comparative bar plot
plt.figure(figsize=(14, 8))
top_n = 15  # Number of top features to show

# Get top features across all models
all_features = set()
for model in shap_results:
    all_features.update(shap_results[model].nlargest(top_n).index.tolist())

# Prepare data for plotting
plot_data = pd.DataFrame()
for feature in all_features:
    for model in shap_results:
        plot_data.loc[feature, model] = shap_results[model].get(feature, 0)

# Normalize to percentage of max importance per model
plot_data = (plot_data / plot_data.max()) * 100

# Sort by total importance
plot_data = plot_data.loc[plot_data.sum(axis=1).sort_values(ascending=False).index]

# Plot
colors = plt.cm.tab20.colors
ax = plot_data.plot(kind='barh', width=0.8, color=colors, figsize=(14, 10))
plt.title('Comparative Feature Importance Across Models', fontsize=14)
plt.xlabel('Normalized SHAP Importance (%)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add value labels for the top 3 models per feature
for i, (idx, row) in enumerate(plot_data.iterrows()):
    top_models = row.nlargest(3)
    for j, (model, val) in enumerate(top_models.items()):
        if val > 5:  # Only label significant values
            ax.text(val + 1, i - 0.1 + j*0.1, f"{model}: {val:.1f}%", 
                   ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'comparative_shap_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All plots and comparison table saved successfully!")