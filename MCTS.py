import torch
import torch.nn.functional as F
from math import sqrt
import numpy as np
from chess_env import ChessEnv
from chess_neural_net import ChessNet
import chess


class MCTSNode:

    def __init__(self, model, state, parent=None, prob=0, move_idx=None) -> None:
        self.visits = 0
        self.value = 0
        self.model = model
        self.state = state
        self.parent = parent
        self.prob = prob
        self.move_idx = move_idx
        self.children = []

    def select(self, c):
        if not self.children or self.visits == 0:
            return None
        
        best_child = (None, -np.inf)
        for child in self.children:
            ucb = self.ucb(child, c)
            if ucb > best_child[1]:
                best_child = (child, ucb)

        return best_child[0]

    @torch.no_grad()
    def expand(self, device):
        encoded_state = AlphaMCTS.env.get_observation(self.state)
        state_tensor = AlphaMCTS.env.move_to_tensor_and_gpu(device, encoded_state)
        state_tensor = state_tensor.unsqueeze(0)
        policy, value = self.model(state_tensor)

        value = value.item()
        policy = F.softmax(policy, dim=-1)
        policy = policy.squeeze(0).detach()
        policy = policy.cpu().numpy() * AlphaMCTS.env.legal_moves_mask(self.state)
        policy /= policy.sum()

        for idx, prob in enumerate(policy):
            if prob > 0:
                move = AlphaMCTS.env.idx_to_move(idx)
                self.state.push(move)
                node = MCTSNode(self.model, self.state, self, prob, idx)
                self.children.append(node)
                self.state.pop()

        return value

    def backpropagate(self, value):
        self.value += value
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)


    def ucb(self, child, c):
        if child.visits == 0:
            q = 0
        else:
            q = 1 - ((child.value / child.visits) + 1) / 2
        return q + c * child.prob * (sqrt(self.visits) / (child.visits + 1))

class AlphaMCTS:

    env = ChessEnv()

    def __init__(self, root, model, device, c=sqrt(2)) -> None:
        self.root = root
        self.model = model
        self.device = device
        self.c = c

    def search(self):
        for _ in range(300):
            node = self.root

            # select
            next_node = node
            while next_node is not None:
                node = next_node
                next_node = node.select(self.c)
        
            # expand
            value = AlphaMCTS.env.get_reward(node.state)
            if value is None:
                value = node.expand(self.device)

            # backpropagate
            node.backpropagate(value)

        action_probs = np.zeros(AlphaMCTS.env.action_space)
        for child in self.root.children:
            action_probs[child.move_idx] = child.visits

        action_probs_sum = np.sum(action_probs)
        if action_probs_sum > 0:
            action_probs /= action_probs_sum
        else:
            return False
        return action_probs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = chess.Board()
    model = ChessNet()
    root = MCTSNode(model, state)
    mcts = AlphaMCTS(root, model, device)

    move = mcts.search()
    move = np.argmax(move)
    move = AlphaMCTS.env.idx_to_move(move)
    state.push(move)
