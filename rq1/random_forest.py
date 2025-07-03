import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

eq5d=pd.read_csv('rq1_cleaned_no_ae.csv')
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = gad7.drop(columns=['SID','GAD7_Round2','GAD7_Round3',
 'GAD7_Round4',
 'GAD7_Round5',
 'GAD7_Round6',
 'GAD7_Round7',
 'GAD7_Round8',
 'GAD7_Round9',
 'GAD7_Round10',
 'GAD7_Round11',
 'GAD7_Round12',
 'GAD7_Round13',
'EQ5D_Round3',
 'EQ5D_Round4',
 'EQ5D_Round5',
 'EQ5D_Round6',
 'GAD7_Round1_x'])

df= df[df['EQ5D_Round2'].notnull()] 

X=df.drop(columns=['EQ5D_Round2'])
y=df['EQ5D_Round2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(random_state=42)


param_grid = {
    'rf__n_estimators': [100, 500, 1000, 2000],
    'rf__max_depth': [None, 5, 10, 20, 50],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4, 8],
    'rf__max_features': ['auto', 'sqrt', 0.2, 0.5],
    'rf__bootstrap': [True, False]
}

pipe = Pipeline([
    ('rf', RandomForestRegressor(random_state=42))
])

# 3. CV splitter
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Instantiate GridSearchCV
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_squared_error',  
    n_jobs=-1,
    verbose=3,
    return_train_score=True
)


grid.fit(X_train, y_train)


results = pd.DataFrame(grid.cv_results_)


for split in ['train', 'test']:
    results[f'mean_{split}_MSE']  = -results[f'mean_{split}_MSE']
    results[f'mean_{split}_RMSE'] = np.sqrt(results[f'mean_{split}_MSE'])


display_cols = [
    'params',
    'mean_train_R2',  'mean_test_R2',
    'mean_train_MSE', 'mean_test_MSE',
    'mean_train_RMSE','mean_test_RMSE'
]
print("Top 5 hyper-parameter settings by CV R²:\n")
print(results[display_cols]
      .sort_values('mean_test_R2', ascending=False)
      .head(5)
      .to_string(index=False))


best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

test_mse  = mean_squared_error(y_test, y_pred)
test_rmse = np.sqrt(test_mse)
test_r2   = r2_score(y_test, y_pred)


print(f"  Test MSE : {test_mse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f}")
print(f"  Test R²  : {test_r2:.4f}")

import os
import joblib
import json

os.makedirs('results', exist_ok=True)


results.to_csv('results/eq5d_round2_rf_cv_results.csv', index=False)


joblib.dump(grid, 'results/eq5d_round2_rf_gridsearch.pkl')


joblib.dump(best_model, 'results/eq5d_round2_rf_best_model.pkl')

test_df = X_test.copy()
test_df['y_true'] = y_test
test_df['y_pred'] = y_pred
test_df.to_csv('results/eq5d_round2_rf_test_predictions.csv', index=False)


metrics = {
    'test_MSE': test_mse,
    'test_RMSE': test_rmse,
    'test_R2': test_r2,
    'best_params': grid.best_params_
}
with open('results/eq5d_round2_rf_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("All results and models saved to the `results/` folder.")
