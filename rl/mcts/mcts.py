"""モンテカルロ木探索 (Monte Carlo Tree Search, MCTS)

AlphaGoで使われた探索アルゴリズム。ゲーム木を効率的に探索する。

4つのフェーズ:
1. 選択 (Selection):    UCB1でノードを選択し、葉ノードに到達
2. 展開 (Expansion):    葉ノードに子ノードを追加
3. シミュレーション (Simulation): ランダムプレイで勝敗を判定
4. 逆伝播 (Backpropagation): 結果を根まで伝播

UCB1:
    UCB1(node) = win_rate + c × √(ln(parent_visits) / node_visits)

TicTacToe（三目並べ）環境で動作確認できる。
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


class TicTacToe:
    """三目並べ環境"""

    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0:空, 1:X, -1:O
        self.current_player = 1

    def clone(self) -> "TicTacToe":
        new = TicTacToe()
        new.board = self.board.copy()
        new.current_player = self.current_player
        return new

    def get_valid_actions(self) -> list[int]:
        return list(np.where(self.board == 0)[0])

    def step(self, action: int) -> tuple[float, bool]:
        """行動を実行し、(報酬, 終了フラグ) を返す"""
        self.board[action] = self.current_player
        winner = self._check_winner()

        if winner != 0:
            return float(winner), True
        if len(self.get_valid_actions()) == 0:
            return 0.0, True  # 引き分け

        self.current_player *= -1
        return 0.0, False

    def _check_winner(self) -> int:
        b = self.board.reshape(3, 3)
        for player in [1, -1]:
            # 行・列
            for i in range(3):
                if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                    return player
            # 対角
            if np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player):
                return player
        return 0


@dataclass
class MCTSNode:
    state: TicTacToe
    parent: Optional["MCTSNode"] = None
    action: Optional[int] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    untried_actions: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.untried_actions:
            self.untried_actions = self.state.get_valid_actions()

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_terminal(self) -> bool:
        return len(self.state.get_valid_actions()) == 0 or self.state._check_winner() != 0

    def ucb1(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float("inf")
        win_rate = self.value / self.visits
        exploration = c * np.sqrt(np.log(self.parent.visits) / self.visits)
        return win_rate + exploration


class MCTS:

    def __init__(self, n_simulations: int = 1000, c: float = 1.41):
        self.n_simulations = n_simulations
        self.c = c

    def search(self, state: TicTacToe) -> int:
        """最善の行動を返す"""
        root = MCTSNode(state=state.clone())

        for _ in range(self.n_simulations):
            node = self._select(root)
            node = self._expand(node)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # 最も訪問回数が多い子を選択
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1で葉ノードまで降りる"""
        while not node.is_terminal and node.is_fully_expanded:
            node = max(node.children, key=lambda c: c.ucb1(self.c))
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """未試行の行動で子ノードを追加"""
        if node.is_terminal or not node.untried_actions:
            return node

        action = node.untried_actions.pop()
        new_state = node.state.clone()
        new_state.step(action)

        child = MCTSNode(state=new_state, parent=node, action=action)
        node.children.append(child)
        return child

    def _simulate(self, node: MCTSNode) -> float:
        """ランダムプレイで結果を得る"""
        state = node.state.clone()

        while True:
            actions = state.get_valid_actions()
            if not actions or state._check_winner() != 0:
                break
            action = np.random.choice(actions)
            state.step(action)

        winner = state._check_winner()
        return float(winner)

    def _backpropagate(self, node: MCTSNode, result: float):
        """結果を根まで伝播"""
        while node is not None:
            node.visits += 1
            # そのノードの手番のプレイヤーから見た価値
            node.value += result * node.state.current_player * -1
            node = node.parent
