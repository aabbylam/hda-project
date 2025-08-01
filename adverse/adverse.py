import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib   

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, make_scorer
from sklearn.neighbors import NearestNeighbors


name = 'any_adverse_binary'
base_dir = '/rds/general/user/hsl121/home/hda_project/adverse/results'
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
df = pd.read_csv('../rq1/rq1_cleaned_adverse_binary.csv')
full['adverse_binary'] = df['adverse_binary']

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

# Separate features and target properly
X = full.drop(columns=drop_cols + ['adverse_binary'])  # Make sure to exclude adverse_binary from features
y = full['adverse_binary']

# Remove rows with missing values
data = pd.concat([X, y], axis=1).dropna()
X_clean = data.drop(columns='adverse_binary')
y_clean = data['adverse_binary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# Debug: Check what y_train actually contains
print(f"y_train type: {type(y_train)}")
print(f"y_train shape: {y_train.shape}")
if hasattr(y_train, 'columns'):
    print(f"y_train columns: {list(y_train.columns)}")
if isinstance(y_train, pd.DataFrame):
    print(f"y_train head:\n{y_train.head()}")
    y_train = y_train['adverse_binary']  # Extract the column if it's a DataFrame

def match_controls_to_cases(cases, controls, match_vars):
    scaler = StandardScaler()
    controls_scaled = scaler.fit_transform(controls[match_vars])
    cases_scaled = scaler.transform(cases[match_vars])

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(controls_scaled)
    _, indices = nn.kneighbors(cases_scaled)

    matched_controls = controls.iloc[indices.flatten()].copy()
    matched_controls['matched_to'] = cases.index.values
    return matched_controls

# Create combined training dataframe with proper index handling
train_combined = pd.DataFrame(X_train)
# Ensure y_train is a Series, not a DataFrame
if isinstance(y_train, pd.DataFrame):
    y_train_series = y_train.iloc[:, 0]  # Take first column if DataFrame
else:
    y_train_series = y_train

train_combined['adverse_binary'] = y_train_series.values  # Use .values to avoid index issues

cases = train_combined[train_combined['adverse_binary'] == 1].reset_index(drop=True)
controls = train_combined[train_combined['adverse_binary'] == 0].reset_index(drop=True)

match_vars = ['Age', 'Sex', 'weight', 'height']
ctrl_sample = controls.sample(frac=1, random_state=0).reset_index(drop=True)
matched_controls = match_controls_to_cases(cases, ctrl_sample, match_vars)
matched_df = pd.concat([cases, matched_controls], ignore_index=True)

X_train_sub = matched_df.drop(columns=['adverse_binary', 'matched_to'], errors='ignore')
y_train_sub = matched_df['adverse_binary']

def get_models_and_grids():
    from sklearn.linear_model import RidgeClassifier
    
    models = {
        'Ridge': Pipeline([('scaler', StandardScaler()), ('model', RidgeClassifier(random_state=42))]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(random_state=42))]),
        'XGB': Pipeline([('scaler', StandardScaler()), ('model', XGBClassifier(random_state=42, eval_metric='logloss'))]),
        'MLP': Pipeline([('scaler', StandardScaler()), ('model', MLPClassifier(random_state=42, max_iter=1000))])
    }

    # Keep your original grid search parameters
    grids = {
        'Ridge': {'model__alpha': [0.1, 1.0, 10.0]}, 
        'Lasso': {'model__C': [0.01, 0.1, 1.0]},     
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
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50), (300, 200, 100), (100, 100, 50, 25)],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam', 'sgd'],
            'model__alpha': [1e-5, 1e-4, 1e-3],
            'model__learning_rate_init': [1e-3, 1e-2],
            'model__batch_size': [32, 64, 128, 256]
        }
    }

    return models, grids

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models, param_grids = get_models_and_grids()

from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

summary = []
detailed_metrics = []

for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    gs = GridSearchCV(model, param_grids[model_name], cv=cv, scoring='f1', n_jobs=-1)
    gs.fit(X_train_sub, y_train_sub)
    best_model = gs.best_estimator_

    # Cross-validation metrics
    auc_scores = cross_val_score(best_model, X_train_sub, y_train_sub, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(best_model, X_train_sub, y_train_sub, cv=cv, scoring='accuracy')

    summary.append({
        'Model': model_name,
        'AUC_Mean': round(auc_scores.mean(), 3),
        'AUC_SD': round(auc_scores.std(), 3),
        'Accuracy_Mean': round(acc_scores.mean(), 3),
        'Accuracy_SD': round(acc_scores.std(), 3),
        'Best_Params': gs.best_params_
    })

    # Detailed classification metrics on best model
    y_pred = best_model.predict(X_train_sub)
    report = classification_report(y_train_sub, y_pred, output_dict=True, zero_division=0)

    detailed_metrics.append({
        'Model': model_name,
        'Precision_0': round(report['0']['precision'], 3),
        'Recall_0': round(report['0']['recall'], 3),
        'F1_0': round(report['0']['f1-score'], 3),
        'Precision_1': round(report['1']['precision'], 3),
        'Recall_1': round(report['1']['recall'], 3),
        'F1_1': round(report['1']['f1-score'], 3),
    })

    # Save model
    joblib.dump(best_model, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

# Save summary of CV performance
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(results_dir, f'{name}_model_metrics_summary.csv'), index=False)

# Save best model classification metrics
detailed_df = pd.DataFrame(detailed_metrics)
detailed_df.to_csv(os.path.join(results_dir, f'{name}_best_model_metrics.csv'), index=False)

# Plot AUC and Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(summary_df['Model']))
width = 0.35

ax.bar(x - width/2, summary_df['AUC_Mean'], width, yerr=summary_df['AUC_SD'], label='AUC', capsize=5)
ax.bar(x + width/2, summary_df['Accuracy_Mean'], width, yerr=summary_df['Accuracy_SD'], label='Accuracy', capsize=5)

ax.set_xticks(x)
ax.set_xticklabels(summary_df['Model'])
ax.set_ylabel("Score")
ax.set_title("Model Performance (5-fold CV)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_model_performance.png'), dpi=300)
plt.close()

