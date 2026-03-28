"""PPO (Proximal Policy Optimization)

REINFORCEの問題（高分散、サンプル効率の悪さ）を解決する方策最適化アルゴリズム。
現代のLLMアライメント（RLHF）でも使われる。

核心的なアイデア: 方策の更新幅を制限する（clipping）。
大きすぎる更新は性能を不安定にするため、新旧方策の比率をクリップする。

目的関数:
    L^CLIP = E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

    r_t = π_new(a|s) / π_old(a|s)   (新旧方策の確率比)
    A_t: アドバンテージ（価値ベースラインとの差）
    ε: クリッピング範囲（通常0.2）

Actor-Critic構成:
    Actor (方策): π(a|s; θ) → 行動の確率分布
    Critic (価値): V(s; φ) → 状態価値の推定
"""

import numpy as np


def _softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class PPO:

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden_dim: int = 32,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        n_update_epochs: int = 4,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.n_update_epochs = n_update_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / hidden_dim)

        # Actor (方策ネットワーク)
        self.actor_W1 = np.random.randn(n_features, hidden_dim) * scale1
        self.actor_b1 = np.zeros(hidden_dim)
        self.actor_W2 = np.random.randn(hidden_dim, n_actions) * scale2
        self.actor_b2 = np.zeros(n_actions)

        # Critic (価値ネットワーク)
        self.critic_W1 = np.random.randn(n_features, hidden_dim) * scale1
        self.critic_b1 = np.zeros(hidden_dim)
        self.critic_W2 = np.random.randn(hidden_dim, 1) * scale2
        self.critic_b2 = np.zeros(1)

    def _actor_forward(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, state @ self.actor_W1 + self.actor_b1)
        logits = h @ self.actor_W2 + self.actor_b2
        return _softmax(logits)

    def _critic_forward(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, state @ self.critic_W1 + self.critic_b1)
        return (h @ self.critic_W2 + self.critic_b2).flatten()

    def select_action(self, state: np.ndarray) -> tuple[int, float]:
        """行動を選択し、その行動の確率も返す"""
        probs = self._actor_forward(state.reshape(1, -1))[0]
        action = int(np.random.choice(self.n_actions, p=probs))
        return action, probs[action]

    def compute_gae(self, rewards, values, dones):
        """GAE (Generalized Advantage Estimation) を計算

        A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
        δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0

        for t in reversed(range(T)):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:T]
        return advantages, returns

    def update(self, states, actions, old_probs, rewards, dones):
        """PPOのパラメータ更新

        states: (T, n_features)
        actions: (T,) int
        old_probs: (T,) 旧方策での行動確率
        rewards: (T,)
        dones: (T,) bool
        """
        states = np.array(states)
        actions = np.array(actions, dtype=int)
        old_probs = np.array(old_probs)
        rewards = np.array(rewards, dtype=np.float64)
        dones = np.array(dones, dtype=np.float64)

        # 価値の推定
        values = self._critic_forward(states)
        advantages, returns = self.compute_gae(rewards, values, dones)

        # アドバンテージの正規化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 複数エポックで更新
        for _ in range(self.n_update_epochs):
            # --- Actor の更新 ---
            h_actor = np.maximum(0, states @ self.actor_W1 + self.actor_b1)
            logits = h_actor @ self.actor_W2 + self.actor_b2
            probs_all = _softmax(logits)
            new_probs = probs_all[np.arange(len(actions)), actions]

            # 確率比
            ratio = new_probs / (old_probs + 1e-10)

            # クリッピングされた目的関数
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -np.mean(np.minimum(surr1, surr2))

            # Actor の勾配計算
            # dL/d(logits) を計算
            clipped = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            use_clipped = (surr2 < surr1).astype(float)
            effective_ratio_grad = (1 - use_clipped) + use_clipped * (
                (ratio >= 1 - self.clip_epsilon) & (ratio <= 1 + self.clip_epsilon)
            ).astype(float)

            d_ratio = -advantages * effective_ratio_grad / len(actions)
            d_new_probs = d_ratio / (old_probs + 1e-10)

            # softmax の勾配
            d_logits = probs_all.copy()
            d_logits[np.arange(len(actions)), actions] -= 1
            d_logits = -d_logits  # log_prob の勾配
            d_logits *= d_new_probs[:, np.newaxis]

            grad_W2_actor = h_actor.T @ d_logits
            grad_b2_actor = np.sum(d_logits, axis=0)
            d_h = d_logits @ self.actor_W2.T * (h_actor > 0)
            grad_W1_actor = states.T @ d_h
            grad_b1_actor = np.sum(d_h, axis=0)

            self.actor_W1 -= self.lr_actor * np.clip(grad_W1_actor, -1, 1)
            self.actor_b1 -= self.lr_actor * np.clip(grad_b1_actor, -1, 1)
            self.actor_W2 -= self.lr_actor * np.clip(grad_W2_actor, -1, 1)
            self.actor_b2 -= self.lr_actor * np.clip(grad_b2_actor, -1, 1)

            # --- Critic の更新 ---
            h_critic = np.maximum(0, states @ self.critic_W1 + self.critic_b1)
            v_pred = (h_critic @ self.critic_W2 + self.critic_b2).flatten()
            critic_loss = np.mean((v_pred - returns) ** 2)

            d_v = 2 * (v_pred - returns) / len(returns)
            grad_W2_critic = h_critic.T @ d_v[:, np.newaxis]
            grad_b2_critic = np.sum(d_v)
            d_h_c = d_v[:, np.newaxis] * self.critic_W2.T * (h_critic > 0)
            grad_W1_critic = states.T @ d_h_c
            grad_b1_critic = np.sum(d_h_c, axis=0)

            self.critic_W1 -= self.lr_critic * np.clip(grad_W1_critic, -1, 1)
            self.critic_b1 -= self.lr_critic * np.clip(grad_b1_critic, -1, 1)
            self.critic_W2 -= self.lr_critic * np.clip(grad_W2_critic, -1, 1)
            self.critic_b2 -= self.lr_critic * np.clip(grad_b2_critic, -1, 1)

        return actor_loss, critic_loss
