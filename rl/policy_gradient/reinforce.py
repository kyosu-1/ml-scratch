"""REINFORCE (Monte Carlo Policy Gradient)

方策勾配法の最も基本的なアルゴリズム。
方策を直接パラメータ化し、期待報酬を最大化するようにパラメータを更新する。

方策: π(a|s) = softmax(sW + b)
目的関数: J(θ) = E[Σ γ^t * r_t]

勾配: ∇J(θ) = E[Σ ∇log π(a_t|s_t) * G_t]
    G_t = Σ_{k=t}^T γ^{k-t} * r_k  (割引累積報酬)

ベースラインとして報酬の平均を引くことで分散を低減:
    ∇J(θ) ≈ Σ ∇log π(a_t|s_t) * (G_t - b)
"""

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class REINFORCE:
    """線形方策を用いたREINFORCE

    状態を特徴量ベクトルに変換する feature_fn を受け取る。
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        lr: float = 0.01,
        gamma: float = 0.99,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma

        self.W = np.zeros((n_features, n_actions))
        self.b = np.zeros(n_actions)

    def _get_probs(self, state_features: np.ndarray) -> np.ndarray:
        logits = state_features @ self.W + self.b
        return _softmax(logits)

    def select_action(self, state_features: np.ndarray) -> int:
        probs = self._get_probs(state_features)
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, episode: list[tuple[np.ndarray, int, float]]):
        """1エピソード分のデータで方策を更新

        episode: [(state_features, action, reward), ...]
        """
        states, actions, rewards = zip(*episode)
        states = np.array(states)
        T = len(rewards)

        # 割引累積報酬を計算
        G = np.zeros(T)
        G[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            G[t] = rewards[t] + self.gamma * G[t + 1]

        # ベースライン (平均報酬)
        baseline = np.mean(G)

        # 勾配計算と更新
        for t in range(T):
            probs = self._get_probs(states[t])
            # ∇log π(a|s) = φ(s) * (1_{a} - π(a|s))
            one_hot = np.zeros(self.n_actions)
            one_hot[actions[t]] = 1
            grad_log = states[t][:, np.newaxis] * (one_hot - probs)

            advantage = G[t] - baseline
            self.W += self.lr * grad_log * advantage
            self.b += self.lr * (one_hot - probs) * advantage
