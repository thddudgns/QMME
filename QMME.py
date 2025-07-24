import os
import glob
import numpy as np
import pandas as pd


def load_quantile_dfs(folder):
    """
    Reads 'SSP1-<model>_EQM_Q{1..5}.csv' files in the given folder
    and returns a dict quantile_dfs where quantile_dfs[q][model] is a DataFrame
    of shape (rows × stations).
    """
    quantile_dfs = {q: {} for q in range(1, 6)}
    for q in range(1, 6):
        pattern = os.path.join(folder, f'SSP1-*_EQM_Q{q}.csv')
        for fp in sorted(glob.glob(pattern)):
            # Extract model name from 'SSP1-<model>_EQM_Q{q}.csv'
            basename = os.path.basename(fp)
            model = basename.replace('SSP1-', '').split('_EQM')[0]
            df = pd.read_csv(fp, encoding='utf-8-sig')
            quantile_dfs[q][model] = df
    return quantile_dfs


def build_and_save_qmme_quantiles(quantile_dfs, weight_df, out_folder):
    """
    Builds the quantile-wise multi-model ensemble (QMME) for each quantile
    and saves the resulting CSV files in out_folder.
    """
    os.makedirs(out_folder, exist_ok=True)

    for q in range(1, 6):
        # Select models that have weights defined
        models = [m for m in quantile_dfs[q] if m in weight_df.index]
        if not models:
            continue

        # Get weight vector for quantile q
        W = weight_df.loc[models, f'Q{q}'].astype(float).values

        # Identify common columns (stations) across all model DataFrames
        dfs = [quantile_dfs[q][m] for m in models]
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        common_cols = sorted(common_cols)

        # Stack data arrays assuming all EQM_Q{q} files have the same number of rows
        arr = np.stack([df[common_cols].values for df in dfs], axis=-1)

        # Compute weighted sum across models (MME)
        mme_arr = np.tensordot(arr, W, axes=([2], [0]))  # result shape: (rows, n_stations)

        mme_df = pd.DataFrame(mme_arr, columns=common_cols)
        # Output filename: SSP1_QMME_Q{q}.csv (adjust if needed)
        outpath = os.path.join(out_folder, f'SSP1_QMME_Q{q}.csv')
        mme_df.to_csv(outpath, index=False, float_format='%.4f', encoding='utf-8-sig')
        print(f"[Q{q}] Saved {mme_arr.shape[0]}×{mme_arr.shape[1]} → {outpath}")


# -----------------------------
# Example execution
# -----------------------------
if __name__ == '__main__':
    quant_folder = r'D:/Korea/QuanNew/tmin'              # Folder containing the modified EQM_Q* files
    weight_path  = r'D:/Korea/SSP1_QMME_MIN_weights.csv'  # Path to the weights CSV file
    out_folder   = r'D:/Korea/QMMe_by_quantile'           # Folder to save the QMME outputs

    quantile_dfs = load_quantile_dfs(quant_folder)
    weight_df    = pd.read_csv(weight_path, index_col=0, encoding='utf-8-sig')
    build_and_save_qmme_quantiles(quantile_dfs, weight_df, out_folder)
