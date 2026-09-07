import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bundle = np.load('prepared_data.npz')
train_inputs = bundle['train_inputs']
train_targets = bundle['train_targets']
val_inputs = bundle['val_inputs']
val_targets = bundle['val_targets']
test_inputs = bundle['test_inputs']
test_targets = bundle['test_targets']
trainval_inputs = np.concatenate([train_inputs, val_inputs], axis=0)
trainval_targets = np.concatenate([train_targets, val_targets], axis=0)
H = int(train_targets.shape[1]) if train_targets.ndim > 1 else 1

feature_scaler = joblib.load('feature_scaler.joblib')
target_scaler = joblib.load('target_scaler.joblib')

train_tensor = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_targets))
val_tensor = TensorDataset(torch.tensor(val_inputs), torch.tensor(val_targets))
test_tensor = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_targets))
train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)
test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)
trainval_loader = DataLoader(TensorDataset(torch.tensor(trainval_inputs), torch.tensor(trainval_targets)),
                             batch_size=64, shuffle=False)

input_dim = train_inputs.shape[-1]
seq_len = train_inputs.shape[1]


class TCNTransformerModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_steps=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, out_steps)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)  


class BiLSTMModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_steps=1):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim * 2, out_steps)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out) 


criterion = nn.MSELoss()


tcn_model = TCNTransformerModel(input_dim, 64, out_steps=H).to(device)
bilstm_model = BiLSTMModel(input_dim, 64, out_steps=H).to(device)

tcn_opt = torch.optim.Adam(tcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
bilstm_opt = torch.optim.Adam(bilstm_model.parameters(), lr=1e-3, weight_decay=1e-4)


best_val = float('inf')
best_state = None
epochs = 20
for epoch in range(epochs):
    tcn_model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(-1)
        preds = tcn_model(xb)
        loss = criterion(preds, yb)
        tcn_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tcn_model.parameters(), 1.0)
        tcn_opt.step()
        train_loss += loss.item()
    tcn_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            preds = tcn_model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
    if val_loss < best_val:
        best_val = val_loss
        best_state = tcn_model.state_dict()
    print(f'TCN epoch {epoch + 1}/{epochs} train {train_loss/len(train_loader):.4f} val {val_loss/len(val_loader):.4f}')
if best_state is not None:
    tcn_model.load_state_dict(best_state)

best_val = float('inf')
best_state = None
for epoch in range(epochs):
    bilstm_model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(-1)
        preds = bilstm_model(xb)
        loss = criterion(preds, yb)
        bilstm_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(bilstm_model.parameters(), 1.0)
        bilstm_opt.step()
        train_loss += loss.item()
    bilstm_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            preds = bilstm_model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
    if val_loss < best_val:
        best_val = val_loss
        best_state = bilstm_model.state_dict()
    print(f'BiLSTM epoch {epoch + 1}/{epochs} train {train_loss/len(train_loader):.4f} val {val_loss/len(val_loader):.4f}')
if best_state is not None:
    bilstm_model.load_state_dict(best_state)


trainval_truth = []
trainval_tcn = []
trainval_bilstm = []
test_truth = []
test_tcn = []
test_bilstm = []

with torch.no_grad():
    tcn_model.eval()
    bilstm_model.eval()
    for xb, yb in trainval_loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(-1)
        pred_tcn = tcn_model(xb)
        pred_bilstm = bilstm_model(xb)
        trainval_truth.append(yb.cpu().numpy())
        trainval_tcn.append(pred_tcn.cpu().numpy())
        trainval_bilstm.append(pred_bilstm.cpu().numpy())
    for xb, yb in test_loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(-1)
        pred_tcn = tcn_model(xb)
        pred_bilstm = bilstm_model(xb)
        test_truth.append(yb.cpu().numpy())
        test_tcn.append(pred_tcn.cpu().numpy())
        test_bilstm.append(pred_bilstm.cpu().numpy())


def inv_transform_multi(arr, scaler):
    arr2 = arr
    if arr2.ndim == 3 and arr2.shape[2] == 1:
        arr2 = arr2.squeeze(-1)
    flat = arr2.reshape(-1, 1)
    inv = scaler.inverse_transform(flat).reshape(arr2.shape)
    return inv

trainval_truth = inv_transform_multi(np.vstack(trainval_truth), target_scaler)
trainval_tcn = inv_transform_multi(np.vstack(trainval_tcn), target_scaler)
trainval_bilstm = inv_transform_multi(np.vstack(trainval_bilstm), target_scaler)

test_truth = inv_transform_multi(np.vstack(test_truth), target_scaler)
test_tcn = inv_transform_multi(np.vstack(test_tcn), target_scaler)
test_bilstm = inv_transform_multi(np.vstack(test_bilstm), target_scaler)
ensemble_trainval = (trainval_tcn + trainval_bilstm) / 2.0
ensemble_test = (test_tcn + test_bilstm) / 2.0

truth_mean = np.mean(test_truth)
sst = np.sum((test_truth - truth_mean) ** 2)

def point_metrics(true_arr, pred_arr):
    ta = np.array(true_arr).flatten()
    pa = np.array(pred_arr).flatten()
    mae = mean_absolute_error(ta, pa)
    mse = mean_squared_error(ta, pa)
    rmse = np.sqrt(mse)
    r2 = 1.0 - np.sum((ta - pa) ** 2) / sst
    return mae, mse, rmse, r2

mae_tcn, mse_tcn, rmse_tcn, r2_tcn = point_metrics(test_truth, test_tcn)
mae_bilstm, mse_bilstm, rmse_bilstm, r2_bilstm = point_metrics(test_truth, test_bilstm)
mae_ens, mse_ens, rmse_ens, r2_ens = point_metrics(test_truth, ensemble_test)

print('--- Point Forecast Metrics (Test) ---')
print(f'TCN     -> MAE {mae_tcn:.3f} MSE {mse_tcn:.3f} RMSE {rmse_tcn:.3f} R2 {r2_tcn:.3f}')
print(f'BiLSTM  -> MAE {mae_bilstm:.3f} MSE {mse_bilstm:.3f} RMSE {rmse_bilstm:.3f} R2 {r2_bilstm:.3f}')
print(f'Ensemble-> MAE {mae_ens:.3f} MSE {mse_ens:.3f} RMSE {rmse_ens:.3f} R2 {r2_ens:.3f}')

np.savez('forecast_results.npz',
         trainval_truth=trainval_truth,
         ensemble_trainval=ensemble_trainval,
         test_truth=test_truth,
         test_tcn=test_tcn,
         test_bilstm=test_bilstm,
         ensemble_test=ensemble_test)
print('预测结果保存到 forecast_results.npz')


plt.figure(figsize=(12, 5))
plt.plot(test_truth.flatten(), label='Real Power', linewidth=2)
plt.plot(test_tcn.flatten(), label='TCN-Transformer', alpha=0.8)
plt.plot(test_bilstm.flatten(), label='BiLSTM', alpha=0.8)
plt.plot(ensemble_test.flatten(), label='Proposed method', linestyle='--', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.tight_layout()
plt.savefig('point_forecast_plot.png', dpi=300)
plt.close()


metrics_dict = {
    'MAE': [mae_tcn, mae_bilstm, mae_ens],
    'RMSE': [rmse_tcn, rmse_bilstm, rmse_ens],
    'MSE': [mse_tcn, mse_bilstm, mse_ens],
    'R2': [r2_tcn, r2_bilstm, r2_ens]
}
model_names = ['TCN', 'BiLSTM', 'Ensemble']
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()
for idx, (name, values) in enumerate(metrics_dict.items()):
    axes[idx].bar(model_names, values, color=['#4c72b0', '#55a868', '#c44e52'])
    axes[idx].set_title(name)
    axes[idx].grid(alpha=0.3, linestyle='--')
plt.suptitle('Point Forecast Metrics')
plt.tight_layout()
plt.savefig('point_metrics_bar.png', dpi=200)
plt.close()
