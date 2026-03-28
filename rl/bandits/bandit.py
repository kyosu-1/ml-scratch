"""多腕バンディット問題 (Multi-Armed Bandit)

探索と活用のトレードオフを扱う問題。

実装:
- Epsilon-Greedy: 確率εでランダム探索、1-εで最良の腕を選択
- UCB (Upper Confidence Bound): 信頼上界を考慮して腕を選択
    UCB = Q(a) + c * √(ln(t) / N(a))
"""

import numpy as np


class EpsilonGreedy:

    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = np.zeros(n_arms)  # 各腕の推定価値
        self.counts = np.zeros(n_arms)     # 各腕の選択回数

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.q_values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        # インクリメンタル平均更新
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class UCB:

    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.q_values = np.zeros(n_arms)
        self.counts = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self) -> int:
        self.total_counts += 1

        # まだ選ばれていない腕があれば優先
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_counts) / self.counts
        )
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class BanditEnvironment:
    """テスト用のバンディット環境"""

    def __init__(self, n_arms: int = 10):
        self.true_values = np.random.randn(n_arms)

    def pull(self, arm: int) -> float:
        return self.true_values[arm] + np.random.randn()
