"""TD学習 (Temporal Difference Learning)

モデルフリーな強化学習アルゴリズム。環境モデルが不要で、経験から直接学習する。

- Q-Learning (Off-policy):
    Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
    行動方策と学習方策が異なる（εグリーディで探索、greedy で学習）

- SARSA (On-policy):
    Q(s,a) ← Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    実際に取った行動 a' を使って更新する
"""

import numpy as np


class QLearning:

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next = np.max(self.q_table[next_state]) if not done else 0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


class SARSA:

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def select_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ):
        next_q = self.q_table[next_state, next_action] if not done else 0
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


class CliffWalkingEnv:
    """Cliff Walking 環境 (4x12 グリッド)

    スタート: (3, 0), ゴール: (3, 11)
    崖: (3, 1) ~ (3, 10) → 報酬 -100, スタートに戻る
    通常の移動: 報酬 -1
    """

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)
        self.goal = (3, 11)
        self.state = self.start

    @property
    def n_states(self):
        return self.rows * self.cols

    @property
    def n_actions(self):
        return 4

    def reset(self) -> int:
        self.state = self.start
        return self._state_to_idx(self.state)

    def step(self, action: int):
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]
        r = max(0, min(self.rows - 1, self.state[0] + dr))
        c = max(0, min(self.cols - 1, self.state[1] + dc))
        self.state = (r, c)

        # 崖に落ちた場合
        if self.state[0] == 3 and 1 <= self.state[1] <= 10:
            self.state = self.start
            return self._state_to_idx(self.state), -100, False

        done = self.state == self.goal
        return self._state_to_idx(self.state), -1, done

    def _state_to_idx(self, state):
        return state[0] * self.cols + state[1]
