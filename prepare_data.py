import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

excel_path = 'autumn.xlsx'
raw_df = pd.read_excel(excel_path).dropna().reset_index(drop=True)
print(f'原始数据行数 {len(raw_df)} 列 {list(raw_df.columns)}')

all_cols = list(raw_df.columns)
if 'Power' not in all_cols:
    raise ValueError('数据里找不到Power列')
target_col = 'Power'
feature_cols = all_cols  


feature_scaler = StandardScaler()
target_scaler = StandardScaler()
feature_values = feature_scaler.fit_transform(raw_df[feature_cols].values)
target_values = target_scaler.fit_transform(raw_df[[target_col]].values)


seq_len = 24
H = 6  
X_rows = []
y_rows = []
for idx in range(len(feature_values) - seq_len - (H - 1)):
    X_rows.append(feature_values[idx:idx + seq_len])
    y_rows.append(target_values[idx + seq_len: idx + seq_len + H])
X_array = np.array(X_rows, dtype=np.float32)
y_array = np.array(y_rows, dtype=np.float32)
print(f'滑动窗口后样本 {len(X_array)} 每个样本形状 {X_array.shape[1:]} 目标形状 {y_array.shape[1:]}')


train_end = int(len(X_array) * 0.7)
val_end = int(len(X_array) * 0.85)

np.savez('prepared_data.npz',
         train_inputs=X_array[:train_end],
         train_targets=y_array[:train_end],
         val_inputs=X_array[train_end:val_end],
         val_targets=y_array[train_end:val_end],
         test_inputs=X_array[val_end:],
         test_targets=y_array[val_end:])


joblib.dump(feature_scaler, 'feature_scaler.joblib')
joblib.dump(target_scaler, 'target_scaler.joblib')

print('数据保存完毕: prepared_data.npz / feature_scaler.joblib / target_scaler.joblib')
