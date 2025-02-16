import chess
from chess import Piece, Move
import numpy as np
import torch


class ChessEnv:

    def __init__(self) -> None:
        self.action_space = 1972
        self._pseudo_board = chess.Board()

        self._knight_moves = [2, 3, 4, 4, 4, 4, 3, 2,
                        3, 4, 6, 6, 6, 6, 4, 3,
                        4, 6, 8, 8, 8, 8, 6, 4, 
                        4, 6, 8, 8, 8, 8, 6, 4, 
                        4, 6, 8, 8, 8, 8, 6, 4, 
                        4, 6, 8, 8, 8, 8, 6, 4, 
                        3, 4, 6, 6, 6, 6, 4, 3, 
                        2, 3, 4, 4, 4, 4, 3, 2]
        
        self._queen_moves = [21, 21, 21, 21, 21, 21, 21, 21,
                         21, 23, 23, 23, 23, 23, 23, 21,
                         21, 23, 25, 25, 25, 25, 23, 21,
                         21, 23, 25, 27, 27, 25, 23, 21,
                         21, 23, 25, 27, 27, 25, 23, 21,
                         21, 23, 25, 25, 25, 25, 23, 21,
                         21, 23, 23, 23, 23, 23, 23, 21,
                         21, 21, 21, 21, 21, 21, 21, 21]
    
    def _create_layer(self, state, piece):
        layer = np.zeros((8, 8), dtype=int)
        piece_map = state.piece_map()

        for square, piece_type in piece_map.items():
            if piece_type == Piece.from_symbol(piece):
                row, col = divmod(square, 8)
                layer[row, col] = 1

        layer = np.flip(layer, axis=0)
        return layer
    
    def _castling_to_idx(self, move):
        moves_idx = {'e1c1' : 1792, 'e1g1' : 1793, 'e8g8' : 1794, 'e8c8' : 1795}
        return moves_idx[move]
    
    def _promotion_with_beating_to_idx(self, move):
        move_idx = 1859
        pieces = {'q' : 1, 'r' : 8, 'n' : 15, 'b' : 22}
        if move[1] == '7':
            move_idx += 56
        
        if ord(move[0]) < ord(move[2]):
            move_idx += 28
            rows = {k : v for v, k in enumerate('abcdefg')}
        else: 
            rows = {k : v for v, k in enumerate('bcdefgh')}

        move_idx += pieces[move[-1]] + rows[move[0]]
        return move_idx

    def _idx_to_promotion_with_beating(self, idx):
        if idx > 1915:
            temp_idx = 1915
            from_col, to_col = 7, 8
        else:
            temp_idx = 1859
            from_col, to_col = 2, 1

        if idx - temp_idx > 28:
            to_row = 1
            temp_idx += 28
            rows = {k : v for k, v in enumerate('abcdefgh')}
        else:
            to_row = -1
            rows = {k-1 : v for k, v in enumerate('abcdefgh')}

        pieces = ((22, 'b'), (15, 'n'), (8, 'r'), (1, 'q'))

        for num, piece in pieces:
            if temp_idx + num > idx:
                continue

            move_piece = piece
            move_row_from = rows[idx - num - temp_idx]
            move_row_to = rows[idx - num - temp_idx + to_row]
            break
        
        return f'{move_row_from}{from_col}{move_row_to}{to_col}{move_piece}'

    def _promotion_to_idx(self, move):
        if move[0] == move[2]:
            pieces = {'q' : 1, 'r' : 9, 'n' : 17, 'b' : 25}
            rows = {k : v for v, k in enumerate('abcdefgh')}
            move_idx = 1795

            move_idx += pieces[move[-1]] + rows[move[0]]

            if move[1] == '7':
                move_idx += 32

            return move_idx
        return self._promotion_with_beating_to_idx(move)


    def _is_move_pseudo_legal(self, move_from, move_target):
        self._pseudo_board.clear_board()
        move = Move(move_from, move_target)
        self._pseudo_board.set_piece_at(move_from, chess.Piece(chess.QUEEN, chess.WHITE))

        if self._pseudo_board.is_pseudo_legal(move):
            return 1
        else:
            self._pseudo_board.remove_piece_at(move_from)
            self._pseudo_board.set_piece_at(move_from, chess.Piece(chess.KNIGHT, chess.WHITE))
            if self._pseudo_board.is_pseudo_legal(move):
                return 1
        return 0

    def _idx_to_castling(self, idx):
        castling_idx = {1792 : 'e1c1', 1793 : 'e1g1', 1794 : 'e8g8', 1795 : 'e8c8'}
        return castling_idx[idx]

    def _idx_to_promotion(self, idx):
        if idx > 1859:
            return self._idx_to_promotion_with_beating(idx)
        
        rows = {k : v for k, v in enumerate('abcdefgh')}

        if idx > 1827:
            pieces = ((1852, 'b'), (1844, 'n'), (1836, 'r'), (1828, 'q'))
        else:
            pieces = ((1820, 'b'), (1812, 'n'), (1804, 'r'), (1796, 'q'))

        for num, piece in pieces:
            if idx < num:
                continue

            move_piece = piece
            move_row = rows[idx - num]
            break
        
        if idx > 1827:
            return f'{move_row}{7}{move_row}{8}{move_piece}'
        return f'{move_row}{2}{move_row}{1}{move_piece}'

    def _idx_to_std_move(self, idx):
        temp_idx = 1791

        for i in range(63, -1, -1):
            temp_idx -= self._knight_moves[i] + self._queen_moves[i]
            if temp_idx < idx:
               move_from = i
               break

        j = self._queen_moves[move_from] + self._knight_moves[move_from]
        k = 63
        while True:
            if self._is_move_pseudo_legal(move_from, k):
                if temp_idx + j == idx:
                    move_target = k
                    break
                j -= 1
            k -= 1

        move = Move(move_from, move_target).uci()
        return move

    def _std_move_to_idx(self, move):
        move = Move.from_uci(move)
        move_from, move_target = move.from_square, move.to_square
        
        move_idx = sum(self._queen_moves[:move_from]) + sum(self._knight_moves[:move_from])
        for i in range(move_target):
            move_idx += self._is_move_pseudo_legal(move_from, i)

        return move_idx

    def idx_to_move(self, idx):
        if idx > 1795:
            move = self._idx_to_promotion(idx)
        elif idx > 1791:
            move = self._idx_to_castling(idx)
        else:
            move = self._idx_to_std_move(idx)
        return Move.from_uci(move)
    
    def _is_castling_legal_move(self, state, move):
        if (move == 'e1g1' and state.has_kingside_castling_rights(chess.WHITE)) or \
            (move == 'e1c1' and state.has_queenside_castling_rights(chess.WHITE)) or \
            (move == 'e8g8' and state.has_kingside_castling_rights(chess.BLACK)) or \
            (move == 'e8c8' and state.has_queenside_castling_rights(chess.BLACK)):
            return True
        return False

    def legal_moves_mask(self, state):
        legal = list(state.legal_moves)
        move_mask = np.zeros(1972, dtype=int)

        for move in legal:
            move = move.uci()
            if self._is_castling_legal_move(state, move):
                move_idx = self._castling_to_idx(move)
            
            elif move[-1] in ('r', 'n', 'b', 'q'):
                move_idx = self._promotion_to_idx(move)

            else:
                move_idx = self._std_move_to_idx(move)

            move_mask[move_idx] = 1

        return move_mask
    
    def move_to_tensor_and_gpu(self, device, *args):
        if len(args) == 1:
            return torch.tensor(args[0], dtype=torch.float32, device=device)
        return [torch.tensor(arr, dtype=torch.float32, device=device) for arr in args if isinstance(arr, np.ndarray)]
            
    def get_observation(self, state):
        layers = []

        for piece in ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']:
            layers.append(self._create_layer(state, piece))
            
        encoded_state = np.stack(layers)

        return encoded_state

    def get_reward(self, state, length=None):
        if state.is_checkmate():
            return 1
            
        if state.is_stalemate() or state.is_insufficient_material() or length == 400:
            return 0
        
        moves_history = list(state.move_stack)
        if len(moves_history) > 8 and (moves_history[-1] == moves_history[-5] == moves_history[-9]):
            return -3
        
        return None
    
if __name__ == '__main__':
    env = ChessEnv()
    for i in range(1860, 1972):
       move = env.idx_to_move(i)
       move = chess.Move.uci(move)
       idx = env._promotion_with_beating_to_idx(move)

       print(f'Iteration: {i}')
       print(f'Moveestimated by env: {move}')
       print(f'Index estimated by env: {idx}\n')