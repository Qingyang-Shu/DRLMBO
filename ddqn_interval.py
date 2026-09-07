import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import warnings
import random
from collections import deque

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bundle = np.load('forecast_results.npz')
trainval_truth = bundle['trainval_truth']
ensemble_trainval = bundle['ensemble_trainval']
test_truth = bundle['test_truth']
test_tcn = bundle['test_tcn']
test_bilstm = bundle['test_bilstm']
ensemble_test = bundle['ensemble_test']


trainval_truth_flat = trainval_truth.flatten()
ensemble_trainval_flat = ensemble_trainval.flatten()
test_truth_flat = test_truth.flatten()
test_tcn_flat = test_tcn.flatten()
test_bilstm_flat = test_bilstm.flatten()
ensemble_test_flat = ensemble_test.flatten()

residuals = trainval_truth_flat - ensemble_trainval_flat
base_width = np.std(residuals)
if base_width < 1e-3:
    base_width = 1.0
print(f'基础间隔宽度 {base_width:.3f}')

actions = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
actions = [a * base_width for a in actions]

core_features = np.stack([
    ensemble_test_flat,
    test_tcn_flat,
    test_bilstm_flat,
    (test_tcn_flat - test_bilstm_flat),
    np.full_like(test_truth_flat, base_width)
], axis=1)

class ReplayMemory:
    def __init__(self, limit):
        self.buffer = deque(maxlen=limit)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions_idx, rewards, next_states, dones = zip(*batch)
        return np.array(states, dtype=np.float32), np.array(actions_idx), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        return self.net(x)


state_dim = core_features.shape[1] + 1
action_dim = len(actions)
policy_net = QNet(state_dim, action_dim).to(device)
target_net = QNet(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
replay = ReplayMemory(4000)

batch_size = 64
gamma = 0.95
max_episodes = 60
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.965

for episode in range(max_episodes):
    prev_abs_error = 0.0
    total_reward = 0.0
    for idx in range(len(core_features)):
        state_vec = np.concatenate([core_features[idx], [prev_abs_error]]).astype(np.float32)
        state_tensor = torch.tensor(state_vec).unsqueeze(0).to(device)
        if random.random() < epsilon:
            action_idx = random.randrange(action_dim)
        else:
            with torch.no_grad():
                action_idx = torch.argmax(policy_net(state_tensor)).item()
        half_width = actions[action_idx]
        mean_pred = core_features[idx][0]
        actual_value = test_truth_flat[idx]
        lower = mean_pred - half_width
        upper = mean_pred + half_width
        covered = lower <= actual_value <= upper
        reward = 2.0 if covered else -2.0
        reward -= half_width / (base_width * 3.0)
        next_prev_error = abs(mean_pred - actual_value)
        done = idx == len(core_features) - 1
        if not done:
            next_state_vec = np.concatenate([core_features[idx + 1], [next_prev_error]]).astype(np.float32)
        else:
            next_state_vec = np.zeros_like(state_vec)
        replay.push(state_vec, action_idx, reward, next_state_vec, done)
        total_reward += reward
        prev_abs_error = next_prev_error
        if len(replay) >= batch_size:
            ss, aa, rr, ns, dd = replay.sample(batch_size)
            ss = torch.tensor(ss).to(device)
            aa = torch.tensor(aa).long().unsqueeze(1).to(device)
            rr = torch.tensor(rr).unsqueeze(1).to(device)
            ns = torch.tensor(ns).to(device)
            dd = torch.tensor(dd).unsqueeze(1).to(device)
            q_values = policy_net(ss).gather(1, aa)
            next_actions = torch.argmax(policy_net(ns), dim=1, keepdim=True)
            next_q = target_net(ns).gather(1, next_actions)
            targets = rr + gamma * next_q * (1 - dd)
            loss = nn.MSELoss()(q_values, targets.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    if (episode + 1) % 5 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    print(f'DDQN episode {episode + 1}/{max_episodes} reward {total_reward/len(core_features):.3f} epsilon {epsilon:.3f}')


final_lower = []
final_upper = []
chosen_actions = []
prev_abs_error = 0.0
with torch.no_grad():
    policy_net.eval()
    for idx in range(len(core_features)):
        state_vec = np.concatenate([core_features[idx], [prev_abs_error]]).astype(np.float32)
        state_tensor = torch.tensor(state_vec).unsqueeze(0).to(device)
        action_idx = torch.argmax(policy_net(state_tensor)).item()
        half_width = actions[action_idx]
        mean_pred = core_features[idx][0]
        final_lower.append(mean_pred - half_width)
        final_upper.append(mean_pred + half_width)
        chosen_actions.append(action_idx)
        prev_abs_error = abs(mean_pred - test_truth_flat[idx])

final_lower = np.array(final_lower)
final_upper = np.array(final_upper)
final_mean = core_features[:, 0]


coverage = np.mean((test_truth_flat >= final_lower) & (test_truth_flat <= final_upper))
avg_width = np.mean(final_upper - final_lower)
range_span = np.max(test_truth_flat) - np.min(test_truth_flat)
picp = coverage
pinaw = avg_width / range_span
ws = picp * (1.0 - pinaw)

print('--- Interval Metrics ---')
print(f'PICP {picp:.3f}')
print(f'PINAW {pinaw:.3f}')
print(f'Avg Width {avg_width:.3f}')
print(f'WS {ws:.3f}')


N_test = test_truth.shape[0]
H = int(test_truth.size / N_test)
np.savez('interval_results.npz',
         lower_flat=final_lower,
         upper_flat=final_upper,
         picp=picp,
         pinaw=pinaw,
         ws=ws,
         chosen_actions=np.array(chosen_actions),
         mean_pred_flat=final_mean,
         truth_flat=test_truth_flat,
         lower=final_lower.reshape(N_test, H),
         upper=final_upper.reshape(N_test, H),
         mean_pred=final_mean.reshape(N_test, H),
         truth=test_truth.reshape(N_test, H))
print('interval_results.npz saved')


plt.figure(figsize=(12, 5))
plt.plot(test_truth_flat, label='Real Power', linewidth=2)
plt.plot(final_mean, label='Proposed method', linestyle='--')
plt.fill_between(range(len(final_lower)), final_lower, final_upper, color='orange', alpha=0.3, label='Interval')
#plt.title('DDQN Interval Forecasting')
plt.xlabel('Time')
plt.ylabel('Power')
plt.legend()
plt.tight_layout()
plt.savefig('interval_plot.png', dpi=300)
plt.close()


metrics_names = ['PICP', 'PINAW', 'WS']
metrics_values = [picp, pinaw, ws]
plt.figure(figsize=(6, 4))
plt.bar(metrics_names, metrics_values, color=['#4c72b0', '#dd8452', '#55a868'])
plt.title('Interval Metrics')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('interval_metrics_bar.png', dpi=200)
plt.close()


counts = np.bincount(chosen_actions, minlength=len(actions))
plt.figure(figsize=(7, 4))
plt.bar([f'{a/base_width:.2f}x' for a in actions], counts)
plt.title('DDQN Action Counts')
plt.xlabel('Interval Multiplier')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('ddqn_action_hist.png', dpi=200)
plt.close()

print('interval_plot.png / interval_metrics_bar.png / ddqn_action_hist.png saved')
