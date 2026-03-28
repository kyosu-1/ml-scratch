"""動的計画法 (Dynamic Programming) による強化学習

環境モデル（遷移確率と報酬）が既知の場合に使えるアルゴリズム。

- 方策反復 (Policy Iteration):
    1. 方策評価: 現在の方策の価値関数を計算
    2. 方策改善: 価値関数に基づき方策を更新
    3. 収束するまで繰り返す

- 価値反復 (Value Iteration):
    V(s) = max_a Σ P(s'|s,a) * [R(s,a,s') + γ * V(s')]
    すべての状態で収束するまで更新

GridWorld環境を使って動作確認できる。
"""

import numpy as np


class GridWorld:
    """4x4 グリッドワールド環境

    状態: 0-15 (4x4のグリッド)
    行動: 0=上, 1=右, 2=下, 3=左
    終了状態: 0, 15
    各遷移の報酬: -1
    """

    def __init__(self, size: int = 4):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.terminal_states = {0, size * size - 1}

    def get_transitions(self, state, action):
        """(next_state, reward, probability) のリストを返す"""
        if state in self.terminal_states:
            return [(state, 0.0, 1.0)]

        row, col = divmod(state, self.size)
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]
        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))
        next_state = new_row * self.size + new_col

        return [(next_state, -1.0, 1.0)]


def value_iteration(env: GridWorld, gamma: float = 1.0, theta: float = 1e-6):
    """価値反復法"""
    V = np.zeros(env.n_states)

    while True:
        delta = 0
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue

            v = V[s]
            action_values = []
            for a in range(env.n_actions):
                q = sum(
                    p * (r + gamma * V[s_next])
                    for s_next, r, p in env.get_transitions(s, a)
                )
                action_values.append(q)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    # 最適方策を導出
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if s in env.terminal_states:
            continue
        action_values = []
        for a in range(env.n_actions):
            q = sum(
                p * (r + gamma * V[s_next])
                for s_next, r, p in env.get_transitions(s, a)
            )
            action_values.append(q)
        policy[s] = np.argmax(action_values)

    return V, policy


def policy_iteration(env: GridWorld, gamma: float = 1.0, theta: float = 1e-6):
    """方策反復法"""
    policy = np.zeros(env.n_states, dtype=int)
    V = np.zeros(env.n_states)

    while True:
        # 方策評価
        while True:
            delta = 0
            for s in range(env.n_states):
                if s in env.terminal_states:
                    continue
                v = V[s]
                a = policy[s]
                V[s] = sum(
                    p * (r + gamma * V[s_next])
                    for s_next, r, p in env.get_transitions(s, a)
                )
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # 方策改善
        policy_stable = True
        for s in range(env.n_states):
            if s in env.terminal_states:
                continue
            old_action = policy[s]
            action_values = []
            for a in range(env.n_actions):
                q = sum(
                    p * (r + gamma * V[s_next])
                    for s_next, r, p in env.get_transitions(s, a)
                )
                action_values.append(q)
            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                policy_stable = False

        if policy_stable:
            break

    return V, policy
