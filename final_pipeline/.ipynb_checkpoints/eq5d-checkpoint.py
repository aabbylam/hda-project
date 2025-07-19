import pandas as pd
import numpy as np
import json
import os


from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor

OUTPUT_DIR = 'results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data(eq5d_path):
    """
    Load preprocessed baseline features and EQ-5D scores (1-, 3-, 6-month).
    Returns feature matrix X and target vector y for Round 2 (3-month).
    """
    df = pd.read_csv(eq5d_path)
    # Expect columns: SID, eq5d_1m, eq5d_3m, eq5d_6m, plus baseline feature columns
    # Set SID as index (not used as feature)
    df = df.set_index('SID')
    df=df[df['EQ5D_Round2'].notna()]
    df=df.dropna(axis=0)

    # Define target and features
    y = df['EQ5D_Round2']
    X = df.drop(columns=['EQ5D_Round2'])
    return X, y

# 2. Define models and grids
def get_models_and_grids():
    models = {
        'Ridge': Ridge(random_state=0),
        'Lasso': Lasso(random_state=0),
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGB': XGBRegressor(objective='reg:squarederror', random_state=42),
        'MLP':MLPRegressor(random_state=42, max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
    }
    grids = {
        'Ridge': {'alpha': [0.01, 0.1, 1, 10, 100]},
        'Lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10]},
        'RandomForest': {'n_estimators': [100, 200, 500 ], 'max_depth': [None, 5, 10, 20],
        'min_samples_split':[2,5,10]},
        'XGB': {'n_estimators': [100, 200, 500], 'max_depth': [2,4,8], 'learning_rate': [0.001,0.01, 0.1]},
        'MLP':{
            'hidden_layer_sizes':[(50,), (100,), (50,50)],
            'alpha':[1e-4, 1e-3, 1e-2],
            'learning_rate_init':[1e-4, 1e-3, 1e-2]}
        
    }
    return models, grids

# 3. Nested CV implementation
def nested_cv(X, y, models, grids):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=2)
    summary = {}

    for name, model in models.items():
        grid = GridSearchCV(
            estimator=model,
            param_grid=grids[name],
            cv=inner_cv,
            scoring='r2',
            n_jobs=-1
        )
        r2_list, mse_list, mae_list = [], [], []
        best_params = None

        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid.fit(X_train, y_train)
            best_params = grid.best_params_

            preds = grid.predict(X_test)
            r2_list.append(r2_score(y_test, preds))
            mse_list.append(mean_squared_error(y_test, preds))
            mae_list.append(mean_absolute_error(y_test, preds))

        summary[name] = {
            'best_params': best_params,
            'r2_mean': np.mean(r2_list),
            'r2_std': np.std(r2_list),
            'mse_mean': np.mean(mse_list),
            'mse_std': np.std(mse_list),
            'mae_mean': np.mean(mae_list),
            'mae_std': np.std(mae_list)
        }
    return summary

# 4. Main execution
def main():
    eq5d_path = '../rq1/rq1_cleaned_no_ae.csv'
    X, y = load_data(eq5d_path)
    models, grids = get_models_and_grids()
    results = nested_cv(X, y, models, grids)

    # Save results
    out_file = os.path.join(OUTPUT_DIR, 'eq5d_round2_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_file}")

if __name__ == '__main__':
    main()
