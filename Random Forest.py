import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from joblib import parallel_backend

# 시나리오 리스트
scenarios = ['SSP245', 'SSP126', 'SSP370', 'SSP585']

for scenario in scenarios:
    inpPath = f''
    output_file_path = f''

    # GCM 파일 불러와서 DataFrame 리스트에 저장
    gcm_files = [f for f in os.listdir(inpPath)
                 if f.startswith('QM_projection_pr_') and f.endswith('.csv')]
    gcm_data_list = []
    for fn in gcm_files:
        path = os.path.join(inpPath, fn)
        df = pd.read_csv(path, encoding='cp949').iloc[:, 1:]  # 첫 컬럼(날짜) 제외
        model_name = os.path.basename(fn).split("_")[0]
        df.columns = [f"{model_name}_{st}" for st in df.columns]
        gcm_data_list.append(df)
    gcm_data = pd.concat(gcm_data_list, axis=1)

    # 관측소 리스트 생성
    station_list = sorted({col.split("_", 1)[1] for col in gcm_data.columns})

    total_days = gcm_data.shape[0]
    predicted_results = pd.DataFrame(index=np.arange(1, total_days + 1))

    # 랜덤 포레스트 하이퍼파라미터 그리드
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }

    for station in station_list:
        feat_cols = [col for col in gcm_data.columns if col.endswith(f"_{station}")]
        X = gcm_data[feat_cols]
        y = X.mean(axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )

        rf_model = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_grid,
            n_iter=30,
            cv=3,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            error_score='raise'
        )
        with parallel_backend('loky', inner_max_num_threads=1):
            search.fit(X_train, y_train)

        best_model = search.best_estimator_
        print(f"[{scenario} - {station}] Best RF params: {search.best_params_}")

        y_test_pred = best_model.predict(X_test)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        print(f"[{scenario} - {station}] Test RMSE = {test_rmse:.4f}")

        # 전체 기간 예측
        y_full_pred = best_model.predict(X)
        y_full_pred = np.clip(y_full_pred, 0, None)
        predicted_results[station] = y_full_pred

    predicted_results.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"Saved Random Forest predictions to {output_file_path}")
