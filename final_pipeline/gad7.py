import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

name = 'gad7_round2'
eq5d=pd.read_csv('../rq1/rq1_cleaned_no_ae.csv')


scores=pd.read_excel('../data/Scores 6 Jan 2025_Prescribed_Completed Baseline PROMs.xlsx')
gad7=scores[scores['promName']=='GAD7']
gad7=gad7[['SID','Round','promName','total_score','completionDate']]


gad7_wide = gad7.pivot_table(index='SID', 
                                    columns='Round', 
                                    values='total_score', 
                                    aggfunc='first')

gad7_wide.columns = [f"GAD7_Round{r}" for r in gad7_wide.columns]
gad7_wide = gad7_wide.reset_index()
gad7= pd.merge(eq5d, gad7_wide, on='SID', how='left')
gad7=gad7.rename(columns={'GAD7_Round1_y': 'GAD7_Round1',})

insomnia=scores[scores['promName']=='insomniaEfficacyMeasure']
insomnia=insomnia[['SID','Round','promName','total_score','completionDate']]


insomnia_wide = insomnia.pivot_table(index='SID', 
                                    columns='Round', 
                                    values='total_score', 
                                    aggfunc='first')

insomnia_wide.columns = [f"insomniaEfficacyMeasure_Round{r}" for r in insomnia_wide.columns]
insomnia_wide = insomnia_wide.reset_index()
insomnia_wide= pd.merge(gad7, insomnia_wide, on='SID', how='left')
insomnia=insomnia_wide.rename(columns={'insomniaEfficacyMeasure_Round1_y': 'insomniaEfficacyMeasure_Round1',})
full=insomnia.copy()


## Splitting into train and test
from sklearn.model_selection import train_test_split

X = full.drop(columns=[
    'SID', 
    'GAD7_Round2', 'GAD7_Round3', 'GAD7_Round4', 'GAD7_Round5',
    'GAD7_Round6', 'GAD7_Round7', 'GAD7_Round8', 'GAD7_Round9',
    'GAD7_Round10', 'GAD7_Round11', 'GAD7_Round12', 'GAD7_Round13',
    'EQ5D_Round2','EQ5D_Round3', 'EQ5D_Round4', 'EQ5D_Round5', 'EQ5D_Round6', 'GAD7_Round1_x','insomniaEfficacyMeasure_Round2',
 'insomniaEfficacyMeasure_Round3',
 'insomniaEfficacyMeasure_Round4',
 'insomniaEfficacyMeasure_Round5',
 'insomniaEfficacyMeasure_Round6',
 'insomniaEfficacyMeasure_Round7',
 'insomniaEfficacyMeasure_Round8',
 'insomniaEfficacyMeasure_Round9',
 'insomniaEfficacyMeasure_Round10',
 'insomniaEfficacyMeasure_Round11',
 'insomniaEfficacyMeasure_Round12',
 'insomniaEfficacyMeasure_Round13'
])
y = full['GAD7_Round2']

data = pd.concat([X, y], axis=1).dropna()
X = data.drop(columns='GAD7_Round2')
y = data['GAD7_Round2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# 2) Define models + grids, wrapping each in a Pipeline that first scales
def get_models_and_grids():
    models = {
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(random_state=0))
        ]),
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(random_state=0))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ]),
        'XGB': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
        ]),
        'MLP': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(random_state=42, max_iter=1000))
        ])
    }

    grids = {
        'Ridge': { 'model__alpha': [0.1, 1.0, 10.0] },
        'Lasso': { 'model__alpha': [0.01, 0.1, 1.0] },

        'RandomForest': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['auto', 'sqrt', 'log2'],
            'model__bootstrap': [True, False]
        },

        'XGB': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 6, 9, 12],
            'model__learning_rate': [0.001, 0.01, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__gamma': [0, 1, 5],
            'model__reg_alpha': [0, 0.01, 0.1],
            'model__reg_lambda': [1, 1.5, 2]
        },

        'MLP': {
            'model__hidden_layer_sizes': [(50,), (100,), (100,50)],
            'model__alpha': [0.0001, 0.001],
            'model__learning_rate_init': [0.001, 0.01]
        }
    }
    return models, grids

models, grids = get_models_and_grids()

# 3) Nested CV
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)

ensemble_results = {}

for name, pipe in models.items():
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grids[name],
        cv=inner_cv,
        scoring='r2',
        n_jobs=-1
    )

    r2_scores, mse_scores = [], []
    for train_idx, test_idx in outer_cv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        gs.fit(X_tr, y_tr)
        preds = gs.predict(X_te)

        r2_scores.append(r2_score(y_te, preds))
        mse_scores.append(mean_squared_error(y_te, preds))

    ensemble_results[name] = {
        'best_params': gs.best_params_,
        'r2_mean': np.mean(r2_scores),
        'r2_std': np.std(r2_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores)
    }

# 4) Save your results:
results_df = pd.DataFrame.from_dict(ensemble_results, orient='index')
results_df.index.name = 'Model'
results_df.reset_index(inplace=True)

results_df.to_csv(f'/rds/general/user/hsl121/home/hda_project/final_pipeline/results/{name}.csv', index=False)
