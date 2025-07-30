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
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib



name = 'sqs_round2'
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
y = full['insomniaEfficacyMeasure_Round2']
data = pd.concat([X, y], axis=1).dropna()
X, y = data.drop(columns='insomniaEfficacyMeasure_Round2'), data['insomniaEfficacyMeasure_Round2']

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
## ranges from 0 to 1

from sklearn.metrics import make_scorer
from sklearn.metrics import cohen_kappa_score

def weighted_kappa(y_true, y_pred):
  try:
    score = cohen_kappa_score(y_true, y_pred, weights='quadratic')
  except:
    score = np.nan
  return score

target_metric = make_scorer(weighted_kappa, greater_is_better=True)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# Define models and parameter grids
def get_models_and_grids():
    models = {
        'Ridge' : OrdinalClassifier(LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)),
        'Lasso' : OrdinalClassifier(LogisticRegression(penalty='l1', solver='saga', C=1.0, max_iter=1000)),
        'RandomForest' : OrdinalClassifier(RandomForestClassifier(random_state=42)),
        'XGB' : OrdinalClassifier(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
        'MLP' : OrdinalClassifier(MLPClassifier(max_iter=1000, random_state=42))
    }
    
    # FIXED: Changed 'model__' to 'clf__' to match OrdinalClassifier parameter name
    grids = {
        'Ridge': {'clf__C': [0.1, 1.0, 10.0]},  # For LogisticRegression, use C instead of alpha
        'Lasso': {'clf__C': [0.01, 0.1, 1.0]},  # For LogisticRegression, use C instead of alpha
        'RandomForest': {
            'clf__n_estimators': [100, 200, 500],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features': ['sqrt', 'log2']  # 'auto' is deprecated, use 'sqrt'
        },
        'XGB': {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [3, 6, 9, 12],
            'clf__learning_rate': [0.001, 0.01, 0.1, 0.2],
            'clf__subsample': [0.6, 0.8, 1.0],
            'clf__colsample_bytree': [0.6, 0.8, 1.0]
        },
        'MLP': {
            'clf__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50), (300, 200, 100), (100, 100, 50, 25)],
            'clf__activation': ['relu', 'tanh'],
            'clf__solver': ['adam', 'sgd'],
            'clf__alpha': [1e-5, 1e-4, 1e-3],
            'clf__learning_rate_init': [1e-3, 1e-2],
            'clf__batch_size': [32, 64, 128, 256]
        }
    }

    return models, grids

models, grids = get_models_and_grids()

# Nested CV to evaluate and select hyperparameters
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)
ensemble_results = {}
fold_kappa = {m: [] for m in models}  # Changed to track kappa instead of r2/mse

for model_name, pipe in models.items():
    print(f"Training {model_name}...")
    gs = GridSearchCV(pipe, grids[model_name], cv=inner_cv, scoring=target_metric, n_jobs=-1)
    kappas = []
    
    for tr_idx, te_idx in outer_cv.split(X):
        gs.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = gs.predict(X.iloc[te_idx])
        kappa = weighted_kappa(y.iloc[te_idx], preds)
        kappas.append(kappa)
        fold_kappa[model_name].append(kappa)
    
    ensemble_results[model_name] = {
        'best_params': gs.best_params_,
        'kappa_mean': np.mean(kappas),
        'kappa_std': np.std(kappas)
    }
    print(f"{model_name} - Kappa: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")

# Save results to CSV
results_df = pd.DataFrame.from_dict(ensemble_results, orient='index')
results_df.index.name = 'Model'
results_df.reset_index(inplace=True)
results_df.to_csv(os.path.join(results_dir, f'{name}_results.csv'), index=False)

# Re-fit each model on full data and save pipeline
for model_name, pipe in models.items():
    best_params = ensemble_results[model_name]['best_params']
    # FIXED: Remove the model__ prefix manipulation since we're using clf__ directly
    final_pipe = pipe.set_params(**best_params)
    final_pipe.fit(X, y)
    joblib.dump(final_pipe, os.path.join(models_dir, f'{name}_{model_name}.pkl'))

# Test set evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report

metrics = []
for model_name in models:
    pipe_path = os.path.join(models_dir, f'{name}_{model_name}.pkl')
    final_pipe = joblib.load(pipe_path)
    preds = final_pipe.predict(X_test)
    
    metrics.append({
        'Model': model_name,
        'Kappa': weighted_kappa(y_test, preds),
        'MAE': mean_absolute_error(y_test, preds),
        'MSE': mean_squared_error(y_test, preds)
    })
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, preds))

metrics_df = pd.DataFrame(metrics)
metrics_csv = os.path.join(fig_dir, f'{name}_model_metrics.csv')
metrics_df.to_csv(metrics_csv, index=False)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Kappa comparison
axes[0].bar(metrics_df['Model'], metrics_df['Kappa'])
axes[0].set_ylabel('Cohen\'s Kappa')
axes[0].set_title(f'{name}: Cohen\'s Kappa by Model')
axes[0].tick_params(axis='x', rotation=45)

# MAE vs MSE
x, w = np.arange(len(metrics_df)), 0.35
axes[1].bar(x - w/2, metrics_df['MAE'], w, label='MAE')
axes[1].bar(x + w/2, metrics_df['MSE'], w, label='MSE')
axes[1].set_xticks(x)
axes[1].set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
axes[1].set_ylabel('Error')
axes[1].set_title(f'{name}: MAE vs MSE by Model')
axes[1].legend()

# CV Kappa distributions
kappa_box = pd.DataFrame(fold_kappa)
kappa_box.plot.box(ax=axes[2])
axes[2].set_ylabel('Cohen\'s Kappa')
axes[2].set_title('CV Kappa Distributions')

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, f'{name}_comprehensive_evaluation.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)