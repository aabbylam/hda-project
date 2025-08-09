import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix, make_scorer
)

# === SETUP ===
name = 'severe_adverse_binary'
base_dir = '/rds/general/user/hsl121/home/hda_project/adverse/results'
results_dir = os.path.join(base_dir, name)
fig_dir = os.path.join(results_dir, 'figures')
models_dir = os.path.join(results_dir, 'models')
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# === LOAD DATA ===
eq5d = pd.read_csv('../rq1/rq1_cleaned_no_ae.csv')
scores = pd.read_excel('../data/Scores 6 Jan 2025_Prescribed_Completed Baseline PROMs.xlsx')
df = pd.read_csv('../rq1/rq1_cleaned_adverse_severe.csv')

# === MERGE PROMs ===
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
full['severe_adverse_binary'] = df['severe_adverse_binary'].astype(int)

# === CLEAN DATA ===
drop_cols = ['SID', 'GAD7_Round2','GAD7_Round3','GAD7_Round4','GAD7_Round5','GAD7_Round6','GAD7_Round7',
             'GAD7_Round8','GAD7_Round9','GAD7_Round10','GAD7_Round11','GAD7_Round12','GAD7_Round13',
             'EQ5D_Round2','EQ5D_Round3','EQ5D_Round4','EQ5D_Round5','EQ5D_Round6',
             'insomniaEfficacyMeasure_Round2','insomniaEfficacyMeasure_Round3','insomniaEfficacyMeasure_Round4',
             'insomniaEfficacyMeasure_Round5','insomniaEfficacyMeasure_Round6','insomniaEfficacyMeasure_Round7',
             'insomniaEfficacyMeasure_Round8','insomniaEfficacyMeasure_Round9','insomniaEfficacyMeasure_Round10',
             'insomniaEfficacyMeasure_Round11','insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13','GAD7_Round1_y', 'insomniaEfficacyMeasure_Round1_y'
]

X = full.drop(columns=drop_cols)
y=full['severe_adverse_binary']

data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='severe_adverse_binary'), data['severe_adverse_binary']
X = X.rename(columns={
    'GAD7_Round1_x': 'GAD7_Round1',
    'insomniaEfficacyMeasure_Round1_x': 'insomniaEfficacyMeasure_Round1'
})

y = data.loc[:, 'severe_adverse_binary'].iloc[:, 0] 

# === CLASS WEIGHTS ===
n_pos = sum(y == 1)
n_neg = sum(y == 0)
class_weights = {0: 1, 1: n_neg/n_pos}

# === MODEL SETUP ===
def get_models_and_grids():
    models = {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l2', solver='liblinear',
                                         class_weight='balanced', random_state=42))
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(penalty='l1', solver='liblinear',
                                         class_weight='balanced', random_state=42))
        ]),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'XGB': XGBClassifier(scale_pos_weight=n_neg/n_pos, random_state=42, eval_metric='logloss'),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(random_state=42, max_iter=1000))
        ])
    }

    param_grids = {
        'Ridge': {'model__C': [0.01, 0.1, 1, 10]},
        'Lasso': {'model__C': [0.01, 0.1, 1, 10]},
        'RandomForest': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'XGB': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.001, 0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        },
        'MLP': {
            'model__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (300, 200, 100)],
            'model__activation': ['relu', 'tanh'],
            'model__solver': ['adam', 'sgd'],
            'model__alpha': [1e-5, 1e-4, 1e-3],
            'model__learning_rate_init': [1e-3, 1e-2],
            'model__batch_size': [32, 64, 128, 256]
        }
    }
    return models, param_grids

# === MODEL SETUP ===
models, param_grids = get_models_and_grids()

# === CV SETUP ===
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = []

# === CV LOOP ===
for model_name in models:
    print(f"\n=== Training {model_name} ===")
    
    outer_scores = {
        'balanced_accuracy': [],
        'auc': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'per_class': [],
        'confusion_matrices': []
    }

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Standard GridSearchCV for all models
        gs = GridSearchCV(models[model_name], param_grids[model_name],
                          scoring=make_scorer(balanced_accuracy_score), cv=inner_cv, n_jobs=-1)
        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)
        y_prob = gs.predict_proba(X_test)[:, 1] if hasattr(gs, "predict_proba") else None

        outer_scores['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
        outer_scores['auc'].append(roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan)
        outer_scores['f1'].append(f1_score(y_test, y_pred))
        outer_scores['precision'].append(precision_score(y_test, y_pred))
        outer_scores['recall'].append(recall_score(y_test, y_pred))
        outer_scores['per_class'].append(classification_report(y_test, y_pred, output_dict=True))
        outer_scores['confusion_matrices'].append(confusion_matrix(y_test, y_pred))

    results.append({
        'Model': model_name,
        'Balanced_Accuracy_Mean': np.mean(outer_scores['balanced_accuracy']),
        'Balanced_Accuracy_Std': np.std(outer_scores['balanced_accuracy']),
        'AUC_Mean': np.mean(outer_scores['auc']),
        'F1_Mean': np.mean(outer_scores['f1']),
        'Precision_Mean': np.mean(outer_scores['precision']),
        'Recall_Mean': np.mean(outer_scores['recall']),
        'Best_Params': gs.best_params_,
        'Per_Class_Metrics': outer_scores['per_class'],
        'Per_Fold_Scores': {
            'balanced_accuracy': outer_scores['balanced_accuracy'],
            'auc': outer_scores['auc']
        },
        'Confusion_Matrices': outer_scores['confusion_matrices'],
        'Mean_Confusion_Matrix': np.mean(outer_scores['confusion_matrices'], axis=0)
    })

    joblib.dump(gs.best_estimator_, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

# === SAVE RESULTS ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, f'{name}_results.csv'), index=False)

# === SAVE AND VISUALIZE CONFUSION MATRICES ===
plt.figure(figsize=(15, 3))
for i, res in enumerate(results):
    plt.subplot(1, 5, i+1)
    cm = res['Mean_Confusion_Matrix']
    
    # Create heatmap
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=['No Adverse', 'Adverse'],
                yticklabels=['No Adverse', 'Adverse'])
    plt.title(f'{res["Model"]}\nMean Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save individual confusion matrices as CSV files
for res in results:
    model_name = res['Model']
    
    # Save mean confusion matrix
    cm_df = pd.DataFrame(res['Mean_Confusion_Matrix'], 
                         index=['Actual_No_Adverse', 'Actual_Adverse'],
                         columns=['Pred_No_Adverse', 'Pred_Adverse'])
    cm_df.to_csv(os.path.join(results_dir, f'{name}_{model_name}_mean_confusion_matrix.csv'))
    
    # Save all fold confusion matrices
    fold_cms = np.array(res['Confusion_Matrices'])
    np.save(os.path.join(results_dir, f'{name}_{model_name}_all_confusion_matrices.npy'), fold_cms)

# === VISUALIZE BALANCED ACCURACY ===
plt.figure(figsize=(10, 6))
x = np.arange(len(results_df))
plt.bar(x, results_df['Balanced_Accuracy_Mean'], yerr=results_df['Balanced_Accuracy_Std'], capsize=10, alpha=0.7)
plt.xticks(x, results_df['Model'], rotation=45, ha='right')
plt.ylabel('Balanced Accuracy')
plt.title('Model Performance Comparison (5-fold Nested CV)')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_performance.png'), dpi=300)
plt.close()


import seaborn as sns

# 1. Prepare long-format data for boxplot
box_data = []

for res in results:
    model = res['Model']
    ba_scores = res['Per_Fold_Scores']['balanced_accuracy']
    auc_scores = res['Per_Fold_Scores']['auc']

    for score in ba_scores:
        box_data.append({'Model': model, 'Metric': 'Balanced Accuracy', 'Value': score})
    for score in auc_scores:
        box_data.append({'Model': model, 'Metric': 'AUC', 'Value': score})


box_df = pd.DataFrame(box_data)

# 2. Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Model', y='Value', hue='Metric', data=box_df)
plt.xticks(rotation=45, ha='right')
plt.title('Boxplot of Balanced Accuracy and AUC per Model')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_boxplot_balacc_auc.png'), dpi=300)
plt.close()