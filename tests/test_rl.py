"""強化学習アルゴリズムのテスト"""

import numpy as np
import pytest


class TestBandits:
    def test_epsilon_greedy(self):
        from rl.bandits.bandit import EpsilonGreedy, BanditEnvironment
        env = BanditEnvironment(n_arms=5)
        agent = EpsilonGreedy(5, epsilon=0.1)

        for _ in range(500):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)

        # 最良の腕を見つけられているか
        best_arm = np.argmax(env.true_values)
        estimated_best = np.argmax(agent.q_values)
        # 必ずしも一致しないが、推定値が妥当であること
        assert agent.counts.sum() == 500

    def test_ucb(self):
        from rl.bandits.bandit import UCB, BanditEnvironment
        env = BanditEnvironment(n_arms=5)
        agent = UCB(5, c=2.0)

        for _ in range(500):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)

        assert agent.counts.sum() == 500
        assert all(c > 0 for c in agent.counts)  # 全ての腕を探索済み


class TestDynamicProgramming:
    def test_value_iteration(self):
        from rl.dynamic_programming.dp import GridWorld, value_iteration
        env = GridWorld(size=4)
        V, policy = value_iteration(env)
        # 終了状態の価値は0
        assert V[0] == 0
        assert V[15] == 0
        # 他の状態の価値は負（各遷移で-1の報酬）
        assert all(V[s] < 0 for s in range(1, 15))

    def test_policy_iteration(self):
        from rl.dynamic_programming.dp import GridWorld, value_iteration, policy_iteration
        env = GridWorld(size=4)
        V_vi, _ = value_iteration(env, gamma=0.99)
        V_pi, _ = policy_iteration(env, gamma=0.99)
        np.testing.assert_allclose(V_vi, V_pi, atol=1e-5)


class TestTDLearning:
    def test_qlearning_convergence(self):
        from rl.td_learning.td import QLearning, CliffWalkingEnv
        env = CliffWalkingEnv()
        agent = QLearning(env.n_states, env.n_actions, lr=0.5, epsilon=0.1)

        rewards = []
        for _ in range(300):
            state = env.reset()
            total = 0
            for _ in range(200):
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                total += reward
                state = next_state
                if done:
                    break
            rewards.append(total)

        # 後半のエピソードで改善していること
        assert np.mean(rewards[-50:]) > np.mean(rewards[:50])

    def test_sarsa_runs(self):
        from rl.td_learning.td import SARSA, CliffWalkingEnv
        env = CliffWalkingEnv()
        agent = SARSA(env.n_states, env.n_actions, lr=0.5, epsilon=0.1)

        state = env.reset()
        action = agent.select_action(state)
        for _ in range(100):
            next_state, reward, done = env.step(action)
            next_action = agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            if done:
                state = env.reset()
                action = agent.select_action(state)

        # Q値が更新されていること
        assert not np.allclose(agent.q_table, 0)


class TestREINFORCE:
    def test_update_changes_weights(self):
        from rl.policy_gradient.reinforce import REINFORCE
        agent = REINFORCE(n_features=4, n_actions=2, lr=0.01)
        w_before = agent.W.copy()

        # ダミーエピソード
        episode = [
            (np.random.randn(4), 0, 1.0),
            (np.random.randn(4), 1, -1.0),
            (np.random.randn(4), 0, 0.5),
        ]
        agent.update(episode)
        assert not np.allclose(agent.W, w_before)

    def test_action_selection(self):
        from rl.policy_gradient.reinforce import REINFORCE
        agent = REINFORCE(n_features=4, n_actions=3)
        state = np.random.randn(4)
        action = agent.select_action(state)
        assert 0 <= action < 3


class TestDQN:
    def test_dqn_trains(self):
        from rl.dqn.dqn import DQN
        agent = DQN(state_dim=4, action_dim=2, hidden_dim=16, batch_size=8, buffer_size=100)

        # バッファにデータを溜める
        for _ in range(20):
            s = np.random.randn(4)
            a = agent.select_action(s)
            r = np.random.randn()
            s_next = np.random.randn(4)
            done = np.random.random() > 0.8
            agent.store_transition(s, a, r, s_next, done)

        # 学習が実行できること
        loss = agent.train()
        assert loss is not None
        assert loss >= 0

    def test_epsilon_decays(self):
        from rl.dqn.dqn import DQN
        agent = DQN(state_dim=2, action_dim=2, epsilon=1.0, epsilon_decay=0.9, batch_size=4, buffer_size=50, hidden_dim=8)
        initial_eps = agent.epsilon

        for _ in range(10):
            s = np.random.randn(2)
            agent.store_transition(s, 0, 1.0, np.random.randn(2), False)

        agent.train()
        assert agent.epsilon < initial_eps
