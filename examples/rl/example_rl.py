"""強化学習アルゴリズムの使用例"""

import numpy as np


def example_bandit():
    print("=== 多腕バンディット ===")
    from rl.bandits.bandit import EpsilonGreedy, UCB, BanditEnvironment

    env = BanditEnvironment(n_arms=10)
    n_steps = 1000

    for name, agent in [
        ("Epsilon-Greedy", EpsilonGreedy(10, epsilon=0.1)),
        ("UCB", UCB(10, c=2.0)),
    ]:
        total_reward = 0
        for _ in range(n_steps):
            arm = agent.select_arm()
            reward = env.pull(arm)
            agent.update(arm, reward)
            total_reward += reward
        print(f"  {name}: 平均報酬 = {total_reward/n_steps:.3f}")


def example_dp():
    print("\n=== 動的計画法 (GridWorld) ===")
    from rl.dynamic_programming.dp import GridWorld, value_iteration, policy_iteration

    env = GridWorld(size=4)
    actions = ["↑", "→", "↓", "←"]

    V_vi, policy_vi = value_iteration(env)
    print("  価値反復法の最適方策:")
    for i in range(4):
        row = [
            actions[policy_vi[i * 4 + j]] if (i * 4 + j) not in env.terminal_states else "G"
            for j in range(4)
        ]
        print(f"    {row}")

    V_pi, policy_pi = policy_iteration(env)
    print(f"\n  価値反復と方策反復の価値関数が一致: {np.allclose(V_vi, V_pi)}")


def example_qlearning():
    print("\n=== Q-Learning (Cliff Walking) ===")
    from rl.td_learning.td import QLearning, SARSA, CliffWalkingEnv

    env = CliffWalkingEnv()
    n_episodes = 500

    for name, agent in [
        ("Q-Learning", QLearning(env.n_states, env.n_actions, lr=0.5, epsilon=0.1)),
        ("SARSA", SARSA(env.n_states, env.n_actions, lr=0.5, epsilon=0.1)),
    ]:
        rewards = []
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0

            if isinstance(agent, SARSA):
                action = agent.select_action(state)

            for _ in range(200):
                if isinstance(agent, QLearning):
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                else:
                    next_state, reward, done = env.step(action)
                    next_action = agent.select_action(next_state)
                    agent.update(state, action, reward, next_state, next_action, done)
                    action = next_action

                total_reward += reward
                state = next_state
                if done:
                    break

            rewards.append(total_reward)

        print(f"  {name}: 最終100エピソードの平均報酬 = {np.mean(rewards[-100:]):.1f}")


if __name__ == "__main__":
    example_bandit()
    example_dp()
    example_qlearning()
