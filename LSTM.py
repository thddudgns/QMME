# -*- coding: utf-8 -*-
# PyTorch LSTM-based time series prediction script (added RMSE saving)
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define LSTM model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.0):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)

# Sequence creation function (using input X and target y)
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.stack(xs), np.stack(ys)

# Main parameters
timesteps = 1  # Using past 5 days of data
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# List of scenarios
scenarios = ['SSP245', 'SSP126', 'SSP370', 'SSP585']

for scenario in scenarios:
    inpPath = f''
    pred_file = f''
    rmse_file = f''

    # Read and merge GCM data
    gcm_files = [f for f in os.listdir(inpPath)
                 if f.startswith('QM_projection_pr_') and f.endswith('.csv')]
    gcm_list = []
    for fn in gcm_files:
        df = pd.read_csv(os.path.join(inpPath, fn), encoding='cp949').iloc[:, 1:]
        model_name = fn.split('_')[0]
        df.columns = [f"{model_name}_{col}" for col in df.columns]
        gcm_list.append(df)
    gcm_data = pd.concat(gcm_list, axis=1)

    station_list = sorted({col.split('_', 1)[1] for col in gcm_data.columns})
    predictions_df = pd.DataFrame(index=np.arange(len(gcm_data)))
    rmse_dict = {}

    for station in station_list:
        # Prepare features and target
        feat_cols = [c for c in gcm_data.columns if c.endswith(f"_{station}")]
        X_raw = gcm_data[feat_cols].values          # shape (N, F)
        y_raw = X_raw.mean(axis=1, keepdims=True)  # shape (N, 1)

        # Create sequences
        X_seq, y_seq = create_sequences(X_raw, y_raw, timesteps)

        # Split into train/test (80/20)
        train_size = int(0.8 * len(X_seq))
        X_train = torch.tensor(X_seq[:train_size], dtype=torch.float32).to(device)
        y_train = torch.tensor(y_seq[:train_size], dtype=torch.float32).to(device)
        X_test = torch.tensor(X_seq[train_size:], dtype=torch.float32).to(device)
        y_test = torch.tensor(y_seq[train_size:], dtype=torch.float32).to(device)

        # DataLoader
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

        # Model, loss function, optimizer
        model = LSTMRegressor(input_dim=X_raw.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(train_loader.dataset)
            if (epoch + 1) % 10 == 0:
                print(f"{scenario}-{station} Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Evaluation and RMSE calculation
        model.eval()
        with torch.no_grad():
            preds_test = model(X_test).cpu().numpy().flatten()
            y_true = y_test.cpu().numpy().flatten()
        rmse = np.sqrt(((preds_test - y_true) ** 2).mean())
        print(f"{scenario}-{station} Test RMSE: {rmse:.4f}")
        rmse_dict[station] = rmse

        # Full-period prediction
        X_full_seq, _ = create_sequences(X_raw, y_raw, timesteps)
        full_X = torch.tensor(X_full_seq, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_full = model(full_X).cpu().numpy().flatten()
        preds_full = np.concatenate([np.full(timesteps, np.nan), preds_full])
        predictions_df[station] = preds_full

    # Save RMSE results
    rmse_df = pd.DataFrame.from_dict(rmse_dict, orient='index', columns=['RMSE'])
    rmse_df.index.name = 'Station'
    rmse_df.to_csv(rmse_file, encoding='utf-8-sig')
    print(f"Saved RMSE results to {rmse_file}")

    # Save prediction results
    predictions_df.to_csv(pred_file, index=False, encoding='utf-8-sig')
    print(f"Saved LSTM predictions for {scenario} to {pred_file}")
