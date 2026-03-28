"""DQN (Deep Q-Network)

Q-LearningをニューラルネットワークQ(s,a;θ)で近似する。

主要な工夫:
1. Experience Replay: 経験をバッファに蓄積し、ランダムにサンプリングして学習
   → データの相関を軽減し、学習を安定化
2. Target Network: Q値の目標計算に使うネットワークを定期的にコピー
   → 学習目標が動くことによる不安定性を軽減

損失: L = E[(r + γ * max_a' Q_target(s',a') - Q(s,a))^2]

DLフレームワークの Linear, ReLU を使って実装。
"""

import numpy as np
from collections import deque
from dl.layers.linear import Linear
from dl.activations.activations import ReLU
from dl.models.sequential import Sequential
from dl.optimizers.optimizers import Adam


class ReplayBuffer:
    """Experience Replay バッファ"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float64),
            np.array(next_states),
            np.array(dones, dtype=np.float64),
        )

    def __len__(self):
        return len(self.buffer)


def _build_network(state_dim: int, action_dim: int, hidden_dim: int = 64):
    return Sequential([
        Linear(state_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, action_dim),
    ])


def _copy_params(src: Sequential, dst: Sequential):
    """src のパラメータを dst にコピー"""
    for s_layer, d_layer in zip(src.layers, dst.layers):
        if hasattr(s_layer, "params_and_grads"):
            for (s_param, _, _), (d_param, _, _) in zip(
                s_layer.params_and_grads(), d_layer.params_and_grads()
            ):
                d_param[:] = s_param


class DQN:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        hidden_dim: int = 64,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Q-Network と Target Network
        self.q_network = _build_network(state_dim, action_dim, hidden_dim)
        self.target_network = _build_network(state_dim, action_dim, hidden_dim)
        _copy_params(self.q_network, self.target_network)

        self.optimizer = Adam(lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.train_step = 0

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.q_network.forward(state.reshape(1, -1))
        return int(np.argmax(q_values[0]))

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # 現在のQ値
        q_values = self.q_network.forward(states)

        # ターゲットQ値
        next_q_values = self.target_network.forward(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        # 選択した行動のQ値に対する勾配
        target_q = q_values.copy()
        for i in range(self.batch_size):
            target_q[i, actions[i]] = targets[i]

        # MSE勾配
        grad = 2 * (q_values - target_q) / self.batch_size
        loss = np.mean((q_values - target_q) ** 2)

        # 逆伝播と更新
        self.q_network.backward(grad)
        self.optimizer.step(self.q_network.layers)

        # ε減衰
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # ターゲットネットワーク更新
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            _copy_params(self.q_network, self.target_network)

        return loss
