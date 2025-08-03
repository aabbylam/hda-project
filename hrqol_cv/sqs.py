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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import seaborn as sns

name = 'sqs_round4'
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
    'insomniaEfficacyMeasure_Round12','insomniaEfficacyMeasure_Round13'
]
X = full.drop(columns=drop_cols)
y = full['insomniaEfficacyMeasure_Round4']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='insomniaEfficacyMeasure_Round4'), data['insomniaEfficacyMeasure_Round4']

## Ordinal Classifier
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.metrics import accuracy_score
class OrdinalClassifier(BaseEstimator):

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                try:
                  clf.module
                except: # For others
                  clf.fit(X, binary_y)
                else: # For MLP
                  binary_y_reshape = binary_y.astype('float32').reshape(-1,1)
                  clf.fit(X, binary_y_reshape)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:,1])
            elif i in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[i-1][:,1] - clfs_predict[i][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i-1][:,1])
        try:
          self.clf.module
        except: # For others
          pred_proba = np.vstack(predicted).T      
        else: # For MLP
          pred_proba = np.hstack((predicted))
        
        return pred_proba

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, sample_weight=None):
        _, indexed_y = np.unique(y, return_inverse=True)
        return accuracy_score(indexed_y, self.predict(X), sample_weight=sample_weight)

## using cohens kappa as the target metric - higher is better
from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score

def weighted_kappa(y_true, y_pred):
  try:
    score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
  except:
    score = np.nan
  return score

# Custom scorer for cross-validation
kappa_scorer = make_scorer(weighted_kappa, greater_is_better=True)

# Define models and parameter grids
def get_models_and_grids():
    models = {
        'Ridge' : Pipeline([('scaler', StandardScaler()), ('model', OrdinalClassifier(LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)))]),
        'Lasso' : Pipeline([('scaler', StandardScaler()), ('model', OrdinalClassifier(LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1000)))]),
        'RandomForest' : Pipeline([('scaler', StandardScaler()), ('model', OrdinalClassifier(RandomForestClassifier(random_state=42)))]),
        'XGB' : Pipeline([('scaler', StandardScaler()), ('model', OrdinalClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))]),
        'MLP' : Pipeline([('scaler', StandardScaler()), ('model', OrdinalClassifier(MLPClassifier(max_iter=1000, random_state=42)))])
    }
    
    grids = {
        'Ridge': {'model__clf__C': [0.1, 1.0, 10.0]},
        'Lasso': {'model__clf__C': [0.01, 0.1, 1.0]},
        'RandomForest': {
            'model__clf__n_estimators': [100, 200, 500],
            'model__clf__max_depth': [None, 5, 10, 20],
            'model__clf__min_samples_split': [2, 5, 10],
            'model__clf__min_samples_leaf': [1, 2, 4],
            'model__clf__max_features': ['sqrt', 'log2']
        },
        'XGB': {
            'model__clf__n_estimators': [100, 200, 300],
            'model__clf__max_depth': [3, 6, 9, 12],
            'model__clf__learning_rate': [0.001, 0.01, 0.1, 0.2],
            'model__clf__subsample': [0.6, 0.8, 1.0],
            'model__clf__colsample_bytree': [0.6, 0.8, 1.0]
        },
        'MLP': {
            'model__clf__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50), (300, 200, 100), (100, 100, 50, 25)],
            'model__clf__activation': ['relu', 'tanh'],
            'model__clf__solver': ['adam', 'sgd'],
            'model__clf__alpha': [1e-5, 1e-4, 1e-3],
            'model__clf__learning_rate_init': [1e-3, 1e-2],
            'model__clf__batch_size': [32, 64, 128, 256]
        }
    }

    return models, grids

cv = KFold(n_splits=5, shuffle=True, random_state=42)
models, grids = get_models_and_grids()
results = []

# Custom scoring functions for classification metrics treated as regression-like
def mse_for_ordinal(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def mae_for_ordinal(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

mse_scorer = make_scorer(mse_for_ordinal, greater_is_better=False)
mae_scorer = make_scorer(mae_for_ordinal, greater_is_better=False)

for model_name, model_pipeline in models.items():
    print(f"Training {model_name}...")
    grid = GridSearchCV(model_pipeline, grids[model_name], cv=cv, scoring=kappa_scorer, n_jobs=-1, refit=True)
    grid.fit(X, y)

    best_model = grid.best_estimator_

    # Cross-validated scores
    kappa_scores = cross_val_score(best_model, X, y, cv=cv, scoring=kappa_scorer)
    neg_mse_scores = cross_val_score(best_model, X, y, cv=cv, scoring=mse_scorer)
    neg_mae_scores = cross_val_score(best_model, X, y, cv=cv, scoring=mae_scorer)

    # Convert negatives and calculate RMSE
    mse_scores = -neg_mse_scores
    rmse_scores = np.sqrt(mse_scores)
    mae_scores = -neg_mae_scores

    results.append({
        'Model': model_name,
        'Best Params': grid.best_params_,
        'Mean Kappa': round(np.mean(kappa_scores), 5),
        'Std Error Kappa': round(np.std(kappa_scores) / np.sqrt(len(kappa_scores)), 5),
        'Mean MSE': round(np.mean(mse_scores), 5),
        'Std Error MSE': round(np.std(mse_scores) / np.sqrt(len(mse_scores)), 5),
        'Mean RMSE': round(np.mean(rmse_scores), 5),
        'Std Error RMSE': round(np.std(rmse_scores) / np.sqrt(len(rmse_scores)), 5),
        'Mean MAE': round(np.mean(mae_scores), 5),
        'Std Error MAE': round(np.std(mae_scores) / np.sqrt(len(mae_scores)), 5),
    })

    # Save best model
    joblib.dump(best_model, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(results_dir, f'{name}_gridsearch_cv_full_metrics.csv'), index=False)

# Boxplots
kappa_box = {}
mse_box = {}

for model_name, model_pipeline in models.items():
    print(f"Collecting fold scores for {model_name}")
    grid = GridSearchCV(model_pipeline, grids[model_name], cv=cv, scoring=kappa_scorer, n_jobs=-1, refit=True)
    grid.fit(X, y)
    best_model = grid.best_estimator_

    # Store per-fold scores
    kappa_scores = cross_val_score(best_model, X, y, cv=cv, scoring=kappa_scorer)
    mse_scores = -cross_val_score(best_model, X, y, cv=cv, scoring=mse_scorer)

    kappa_box[model_name] = kappa_scores
    mse_box[model_name] = mse_scores

# === Convert to DataFrames ===
kappa_df = pd.DataFrame(kappa_box)
mse_df = pd.DataFrame(mse_box)

# === Plot Kappa Boxplot ===
plt.figure(figsize=(10, 5))
sns.boxplot(data=kappa_df)
plt.title('Cross-Validated Cohen\'s Kappa by Model')
plt.ylabel('Cohen\'s Kappa Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'{name}_kappa_boxplot.png'), dpi=300)
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