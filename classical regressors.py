import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import pairwise_distances

# Configurations
scenarios = ['SSP245', 'SSP126', 'SSP370', 'SSP585']
n_neighbors = 5  # for KNN
poly_degree = 2  # for Polynomial Regression

for scenario in scenarios:
    print(f"\n*** Starting scenario: {scenario} ***")
    inp_path = f'/home/visionlab01/young/MLP/{scenario}'
    # Use a writable output directory under the user home
    output_dir = os.path.expanduser(f'~/MLP_comparison/{scenario}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load GCM data
    gcm_files = [f for f in os.listdir(inp_path) if f.startswith('QM_projection_pr_') and f.endswith('.csv')]
    print(f"Found {len(gcm_files)} GCM files: {gcm_files}")
    gcm_data_list = []
    for fn in gcm_files:
        file_path = os.path.join(inp_path, fn)
        print(f"  Loading file: {fn}")
        df = pd.read_csv(file_path, encoding='cp949')
        print(f"    Raw shape: {df.shape}")
        df = df.iloc[:, 1:]
        model_name = os.path.basename(fn).split("_")[0]
        df.columns = [f"{model_name}_{st}" for st in df.columns]
        print(f"    Renamed columns example: {df.columns[:3].tolist()}")
        gcm_data_list.append(df)
    gcm_data = pd.concat(gcm_data_list, axis=1)
    print(f"Combined data shape: {gcm_data.shape}")

    station_list = sorted({col.split("_",1)[1] for col in gcm_data.columns})
    print(f"Processing {len(station_list)} stations, example: {station_list[:3]}")

    # Prepare storage DataFrames
    df_mlr = pd.DataFrame(index=gcm_data.index)
    df_poly = pd.DataFrame(index=gcm_data.index)
    df_knn = pd.DataFrame(index=gcm_data.index)
    df_idw = pd.DataFrame(index=gcm_data.index)
    rmse_records = []

    # Loop through stations
    for station in station_list:
        print(f"\n-- Station: {station} --")
        feat_cols = [c for c in gcm_data.columns if c.endswith(f"_{station}")]
        X = gcm_data[feat_cols]
        y = X.mean(axis=1)
        print(f"  Features shape: {X.shape}, target shape: {y.shape}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        print(f"  Train/Test sizes: {X_train.shape}/{X_test.shape}")

        # 1) MLR
        mlr = LinearRegression().fit(X_train, y_train)
        y_mlr_pred = mlr.predict(X_test)
        mlr_rmse = mean_squared_error(y_test, y_mlr_pred, squared=False)
        print(f"    MLR test RMSE: {mlr_rmse:.4f}")
        df_mlr[station] = mlr.predict(X)

        # 2) Polynomial Regression
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_poly = poly.transform(X)
        poly_model = Ridge(alpha=1.0).fit(X_train_poly, y_train)
        y_poly_pred = poly_model.predict(poly.transform(X_test))
        poly_rmse = mean_squared_error(y_test, y_poly_pred, squared=False)
        print(f"    Poly deg={poly_degree} test RMSE: {poly_rmse:.4f}")
        df_poly[station] = poly_model.predict(X_poly)

        # 3) KNN
        knn = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
        y_knn_pred = knn.predict(X_test)
        knn_rmse = mean_squared_error(y_test, y_knn_pred, squared=False)
        print(f"    KNN k={n_neighbors} test RMSE: {knn_rmse:.4f}")
        df_knn[station] = knn.predict(X)

        # 4) IDW
        dists_full = pairwise_distances(X, X_train)
        weights_full = 1 / (dists_full + 1e-6)
        idw_full = (weights_full * y_train.values).sum(axis=1) / weights_full.sum(axis=1)
        idw_rmse = mean_squared_error(y_test, idw_full[y_test.index], squared=False)
        print(f"    IDW test RMSE: {idw_rmse:.4f}")
        df_idw[station] = idw_full

        # Collect RMSE
        rmse_records.append({
            'station': station,
            'MLR_RMSE': mlr_rmse,
            'Poly_RMSE': poly_rmse,
            'KNN_RMSE': knn_rmse,
            'IDW_RMSE': idw_rmse
        })

    # Save predictions
    df_mlr.to_csv(os.path.join(output_dir, f"{scenario}_MLR_predictions.csv"), encoding='utf-8-sig')
    df_poly.to_csv(os.path.join(output_dir, f"{scenario}_Poly_predictions.csv"), encoding='utf-8-sig')
    df_knn.to_csv(os.path.join(output_dir, f"{scenario}_KNN_predictions.csv"), encoding='utf-8-sig')
    df_idw.to_csv(os.path.join(output_dir, f"{scenario}_IDW_predictions.csv"), encoding='utf-8-sig')
    print("\nSaved full-series predictions.")

    # Save RMSE summary
    rmse_df = pd.DataFrame(rmse_records)
    rmse_df.to_csv(os.path.join(output_dir, f"{scenario}_test_RMSE_summary.csv"), index=False, encoding='utf-8-sig')
    print(f"Saved RMSE summary to {os.path.join(output_dir, f'{scenario}_test_RMSE_summary.csv')}")
