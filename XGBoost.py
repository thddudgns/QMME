import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from joblib import parallel_backend

# List of scenarios
scenarios = ['SSP245', 'SSP126', 'SSP370', 'SSP585']

for scenario in scenarios:
    inpPath = f'D:/Bias correction/{scenario}'
    output_file_path = f'D:/Bias correction/{scenario}/XGBoost_Optimized_Results_{scenario}.csv'

    # Load GCM files into a list of DataFrames
    gcm_files = [f for f in os.listdir(inpPath)
                 if f.startswith('QM_projection_pr_') and f.endswith('.csv')]
    gcm_data_list = []
    for fn in gcm_files:
        path = os.path.join(inpPath, fn)
        df = pd.read_csv(path, encoding='cp949').iloc[:, 1:]  # exclude first column (date)
        # Rename columns to modelName_station format
        model_name = os.path.basename(fn).split("_")[0]
        df.columns = [f"{model_name}_{st}" for st in df.columns]
        gcm_data_list.append(df)
    # Concatenate all 14 GCM DataFrames into one
    gcm_data = pd.concat(gcm_data_list, axis=1)

    # Create station list (suffix of each column name)
    station_list = sorted({col.split("_", 1)[1] for col in gcm_data.columns})

    # Prepare DataFrame for saving results and dict for residuals
    total_days = gcm_data.shape[0]
    predicted_results = pd.DataFrame(index=np.arange(1, total_days + 1))
    residuals_dict = {}

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [300, 500, 700],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.001, 0.005, 0.01, 0.05],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'min_child_weight': [1, 3, 5]
    }

    for station in station_list:
        # Select the 14 GCM columns for the current station
        feat_cols = [col for col in gcm_data.columns if col.endswith(f"_{station}")]
        X = gcm_data[feat_cols]
        y = X.mean(axis=1)

        # Train/test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        # Define model and set up RandomizedSearchCV
        xgb_model = xgb.XGBRegressor(random_state=42, eval_metric='rmse')
        search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=50,
            cv=3,
            n_jobs=32,
            verbose=2,
            scoring='neg_mean_squared_error',
            error_score='raise'
        )
        with parallel_backend('loky', inner_max_num_threads=1):
            search.fit(X_train.values, y_train.values)

        best_model = search.best_estimator_
        print(f"[{scenario} - {station}] Best params: {search.best_params_}")

        # Predict on test set
