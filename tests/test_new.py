"""新規追加アルゴリズムのテスト"""

import numpy as np
import pytest


class TestGMM:
    def test_clustering(self):
        from ml.mixture.gmm import GaussianMixture
        rng = np.random.RandomState(42)
        c1 = rng.randn(30, 2) + np.array([0, 0])
        c2 = rng.randn(30, 2) + np.array([5, 5])
        X = np.vstack([c1, c2])

        gmm = GaussianMixture(n_components=2, n_iters=50)
        gmm.fit(X)

        assert len(np.unique(gmm.labels_)) == 2
        assert gmm.weights_.shape == (2,)
        np.testing.assert_allclose(gmm.weights_.sum(), 1.0, atol=1e-5)

    def test_predict_proba(self):
        from ml.mixture.gmm import GaussianMixture
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        gmm = GaussianMixture(n_components=2).fit(X)
        proba = gmm.predict_proba(X)
        # 各行の合計が1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestWord2Vec:
    def test_training(self):
        from dl.nlp.word2vec import Word2Vec
        corpus = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "ran", "in", "the", "park"],
            ["the", "cat", "and", "the", "dog"],
        ] * 10

        model = Word2Vec(embed_dim=10, window_size=2, n_epochs=3)
        model.fit(corpus)

        assert model.vocab_size > 0
        vec = model.word_vector("cat")
        assert vec.shape == (10,)

    def test_most_similar(self):
        from dl.nlp.word2vec import Word2Vec
        corpus = [["a", "b", "c", "d"]] * 20
        model = Word2Vec(embed_dim=8, n_epochs=3).fit(corpus)
        results = model.most_similar("a", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[1], float) for r in results)


class TestVAE:
    def test_forward_backward(self):
        from dl.models.vae import VAE
        vae = VAE(input_dim=8, hidden_dim=16, latent_dim=2)
        x = np.random.rand(4, 8)
        x_recon, mu, log_var, z = vae.forward(x)

        assert x_recon.shape == (4, 8)
        assert mu.shape == (4, 2)
        assert z.shape == (4, 2)

        loss, recon_loss, kl_loss = vae.loss(x, x_recon, mu, log_var)
        assert loss >= 0

        vae.backward(x, x_recon, mu, log_var)
        assert vae.grad_W_enc1 is not None

    def test_generate(self):
        from dl.models.vae import VAE
        vae = VAE(input_dim=8, hidden_dim=16, latent_dim=2)
        samples = vae.generate(n_samples=5)
        assert samples.shape == (5, 8)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_loss_decreases(self):
        from dl.models.vae import VAE
        from dl.optimizers.optimizers import Adam
        vae = VAE(input_dim=4, hidden_dim=8, latent_dim=2)
        x = np.random.rand(20, 4)
        opt = Adam(lr=1e-3)

        losses = []
        for _ in range(30):
            x_recon, mu, log_var, z = vae.forward(x)
            loss, _, _ = vae.loss(x, x_recon, mu, log_var)
            vae.backward(x, x_recon, mu, log_var)
            opt.step([vae])
            losses.append(loss)

        assert losses[-1] < losses[0]


class TestGAN:
    def test_train_step(self):
        from dl.models.gan import GAN
        gan = GAN(data_dim=4, latent_dim=8, hidden_dim=16)
        real_data = np.random.randn(8, 4)

        d_loss, g_loss = gan.train_step(real_data)
        assert isinstance(d_loss, float)
        assert isinstance(g_loss, float)

    def test_generate(self):
        from dl.models.gan import GAN
        gan = GAN(data_dim=4, latent_dim=8, hidden_dim=16)
        samples = gan.generate(n_samples=5)
        assert samples.shape == (5, 4)


class TestDiffusion:
    def test_train_step(self):
        from dl.models.diffusion import DDPM
        ddpm = DDPM(data_dim=4, n_timesteps=50, hidden_dim=32)
        x = np.random.randn(8, 4)

        loss = ddpm.train_step(x)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_sample(self):
        from dl.models.diffusion import DDPM
        ddpm = DDPM(data_dim=4, n_timesteps=10, hidden_dim=16)
        samples = ddpm.sample(n_samples=3)
        assert samples.shape == (3, 4)

    def test_loss_decreases(self):
        from dl.models.diffusion import DDPM
        rng = np.random.RandomState(42)
        data = rng.randn(50, 2) * 0.5 + np.array([2, 2])

        ddpm = DDPM(data_dim=2, n_timesteps=20, hidden_dim=32, lr=1e-3)
        losses = []
        for _ in range(50):
            idx = rng.choice(len(data), 16)
            loss = ddpm.train_step(data[idx])
            losses.append(loss)

        assert np.mean(losses[-10:]) < np.mean(losses[:10])


class TestPPO:
    def test_select_action(self):
        from rl.policy_gradient.ppo import PPO
        agent = PPO(n_features=4, n_actions=2)
        state = np.random.randn(4)
        action, prob = agent.select_action(state)
        assert 0 <= action < 2
        assert 0 < prob <= 1

    def test_gae(self):
        from rl.policy_gradient.ppo import PPO
        agent = PPO(n_features=4, n_actions=2)
        rewards = [1.0, 0.0, 1.0, 0.0, 1.0]
        values = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        dones = [False, False, False, False, True]
        advantages, returns = agent.compute_gae(rewards, values, dones)
        assert len(advantages) == 5
        assert len(returns) == 5

    def test_update(self):
        from rl.policy_gradient.ppo import PPO
        agent = PPO(n_features=4, n_actions=2, n_update_epochs=2)
        states = [np.random.randn(4) for _ in range(10)]
        actions = [np.random.randint(2) for _ in range(10)]
        old_probs = [0.5] * 10
        rewards = [1.0] * 10
        dones = [False] * 9 + [True]

        actor_loss, critic_loss = agent.update(states, actions, old_probs, rewards, dones)
        assert isinstance(actor_loss, float)
        assert isinstance(critic_loss, float)


class TestMCTS:
    def test_tictactoe(self):
        from rl.mcts.mcts import TicTacToe
        env = TicTacToe()
        assert len(env.get_valid_actions()) == 9
        _, done = env.step(0)
        assert not done
        assert len(env.get_valid_actions()) == 8

    def test_search(self):
        from rl.mcts.mcts import MCTS, TicTacToe
        env = TicTacToe()
        mcts = MCTS(n_simulations=100)
        action = mcts.search(env)
        assert action in env.get_valid_actions()

    def test_blocking_move(self):
        """MCTSが相手の勝ちを防ぐ手を選べるか"""
        from rl.mcts.mcts import MCTS, TicTacToe
        env = TicTacToe()
        # X: 0, 1 → Oはブロックすべき (2に打つ)
        env.step(0)  # X plays 0
        env.step(3)  # O plays 3
        env.step(1)  # X plays 1 → X is about to win at 2

        mcts = MCTS(n_simulations=500)
        action = mcts.search(env)  # O's turn
        assert action == 2, f"MCTSがブロックしなかった: {action}"
