import numpy as np
import chess
import os
import pickle
import torch
from time import perf_counter
from collections import deque
from torch.optim import AdamW
import torch.nn.functional as F
import torch.multiprocessing as mp
from MCTS import AlphaMCTS, MCTSNode
from chess_neural_net import ChessNet


class SelfPlayLearning:

    def __init__(self, model, optimizer, device, batch_size, learning_cycles, games_in_one_cycle=20, epochs=32) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.learning_cycles = learning_cycles
        self.games_in_one_cycle = games_in_one_cycle
        self.epochs = epochs
        self.model.to(self.device)
        self.history = deque(maxlen=2000)
        
        self.initial_c = 2
        self.final_c = 1.4
        self.c_factor = (self.final_c / self.initial_c) ** (1 / 3000)

    def _set_c_param(self, game_idx):
        return self.initial_c * (self.c_factor ** game_idx)

    def _load_params_and_get_game_idx(self):
        folder = 'weights'
        if not os.path.exists(folder) or (os.path.isdir(folder) and not os.listdir(folder)):
            return 1
        
        files = [os.path.join(folder, name) for name in os.listdir(folder)]
        file_name = max(files, key=os.path.getmtime)

        game_idx = file_name.rfind('_')
        game_idx = file_name[game_idx + 1 : -3]

        weights = torch.load(file_name)
        self.model.load_state_dict(weights['model_state_dict'])
        self.optimizer.load_state_dict(weights['optimizer_state_dict'])

        return int(game_idx) + 1
    
    def _save_model_and_optimizer(self, i):
        os.makedirs('weights', exist_ok=True)
        save_path = os.path.join('weights', f'model_and_optimizer_{i-1}.pt')

        torch.save({
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict()},
            save_path)
        
    def _save_history(self):
        with open('games_history.pkl', 'wb') as file:
            pickle.dump(self.history, file)

    def _load_history(self):
        if not os.path.exists('games_history.pkl'):
            return deque(maxlen=2000)
        
        with open('games_history.pkl', 'rb') as file:
            history = pickle.load(file)
        return history

    def _start_games(self, game_idx, shared_lst):
        c = self._set_c_param(game_idx)

        with mp.Pool(processes=5) as pool:
            results = pool.starmap(self._play_game, [(shared_lst, c) for _ in range(self.games_in_one_cycle)])

        for _ in range(self.games_in_one_cycle):
            self.history.append(shared_lst.pop())

        print('All processes in cycle done')

    def _play_game(self, shared_lst, c):
        torch.device(self.device)
        game_data = []
        state = chess.Board()
        root = MCTSNode(self.model, state)
        mcts = AlphaMCTS(root, self.model, self.device, c)

        while True:
            policy = mcts.search()

            if not isinstance(policy, bool):
                encoded_state = AlphaMCTS.env.get_observation(state)
                game_data.append([encoded_state, policy])
                move  = np.random.choice(AlphaMCTS.env.action_space, p=policy)
                move = AlphaMCTS.env.idx_to_move(move)
                state.push(move)
                root = MCTSNode(self.model, state)
                mcts.root = root
                result = AlphaMCTS.env.get_reward(state, len(game_data))
            else:
                result = -1
        
            if result is not None:
                print(result)
                if result == -3:
                    n = len(game_data) - 7
                    game_data[n : ] = [x + [-1] for x in game_data[n : ]]
                    game_data[ : n] = [x + [0] for x in game_data[: n]]

                else:
                    for idx in range(len(game_data) - 1, -1, -1):
                        game_data[idx].append(result)
                        result = -result
                break
        
        shared_lst.append(game_data)
        print(f'Moves done: {len(game_data)}\n')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def learn(self):
        game_idx = self._load_params_and_get_game_idx()
        print(game_idx)
        self.history = self._load_history()
        
        with mp.Manager() as menager:
            shared_lst = menager.list()

            for i in range(self.learning_cycles):
                self.model.eval()
                print(f'Game index: {game_idx} - {game_idx + self.games_in_one_cycle - 1}\n')

                start = perf_counter()
                self._start_games(game_idx, shared_lst)
                end = perf_counter()

                print(f'Model played games in {end - start} sec\n')

                self.model.train()
                for _ in range(self.epochs):
                    self.train()

                game_idx += self.games_in_one_cycle
                self._save_model_and_optimizer(game_idx)
                self._save_history()
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
                
    def train(self):
        data = [x for game in self.history for x in game]
        np.random.shuffle(data)
        n = len(data)

        for batch_idx in range(0, n, self.batch_size):
            sample = data[batch_idx : min(batch_idx + self.batch_size, n)]

            # pogrupować dane na 3 tablice states, policy_target, value_target
            states, policy_targets, value_targets = zip(*sample)
            # przekształcić na tensory
            states, policy_targets, value_targets = np.array(states), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            # Różnica wartości dla wygranej od faktycznego wyniku
            result_loss = sum((value_targets - 1)**2)

            states, policy_targets, value_targets, result_loss = AlphaMCTS.env.move_to_tensor_and_gpu(self.device, states, policy_targets, value_targets, result_loss)

            # dokonać predykcji modelu
            policy_pred, value_pred = self.model(states)

            # obliczyć cross entropy loss dla policy
            policy_loss = F.cross_entropy(policy_pred, policy_targets)

            # obliczyć mean squared error dla value
            value_loss = F.mse_loss(value_pred, value_targets)

            # całkowita strata
            total_loss = policy_loss + value_loss + result_loss

            # backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    device = torch.device('cuda') 
    state = chess.Board()
    model = ChessNet()
    optimizer = AdamW(model.parameters(), lr=0.001, fused=True, weight_decay=1e-5)
    chess_agent = SelfPlayLearning(model, optimizer, device, batch_size=128, learning_cycles=1, games_in_one_cycle=20, epochs=32)
    start = perf_counter()
    chess_agent.learn()
    end = perf_counter()
    print(f'Total learning time: {end - start}')