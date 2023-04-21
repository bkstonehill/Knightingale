import sys

import chess
import chess.pgn
import chess.svg
import chess.engine
from BRVIT import BRViT
from IPython.display import SVG

import random
import time
from datetime import datetime
import os

import math
import numpy as np

import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def fen2img(fen: str):
    '''
    Utility function to use for evaluations and hashing. Converts chess board into a 8x8x1 image with values in
    the range of [-6, 6] representing the chess pieces.

    Parameters:
        fen: 'str' - string representation of the board

    Returns -> 'numpy.ndarray' - image representation of the board
    '''

    mapping = {
        'P': 1,
        'N': 2,
        'B': 3,
        'R': 4,
        'Q': 5,
        'K': 6,
        'p': -1,
        'n': -2,
        'b': -3,
        'r': -4,
        'q': -5,
        'k': -6,
        ' ': 0
    }

    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k', ' ']
    numbers = range(1, 9)

    string_brd = fen.split(' ')[0]
    for num in numbers:
        string_brd = string_brd.replace(str(num), num * ' ')

    img = np.zeros((8, 8))

    for i, row in enumerate(string_brd.split('/')):
        for j, square in enumerate(row):
            img[i, j] = mapping[square]

    return torch.from_numpy(img.reshape((1, 8, 8)).astype(np.float32))


class Knightingale:
    def __init__(self, table_size: int, is_white: bool, board: chess.Board):
        """
        A deep learning chess engine that leverages the architectures of Block-Recurrent Transformers (Hutchins, D.,
        Schlag, I., Wu, Y., Dyer, E., & Neyshabur, B. (2022). Block-Recurrent Transformers. ArXiv. /abs/2203.07852) with
        Vision Transformers (ViT) (Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
        Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly,
        Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale.
        In ICLR, 2021.)

        The engine utilizes the hybrid transformer architecture to evaluate positions and is trained upon completed
        chess games. Instead of using genetic algorithms or reinforcment learning to train the model, it is trained
        through recursive backpropagation based upon the difference between the actual score of the game (-1 for black,
        0 for draw, and 1 for white) with the predicted score.

        Parameters:
            table_size: 'int' - The size of the transposition table for storing positions for faster computation

            is_white: 'bool'  - Flag indicating if the engine is the first or second player (Affects evaluation
                                comparisions)

            board: 'chess.Board' - A copy of the starting board state (Used for internal move operations when searching)
        """

        self.TABLE_LIMIT = table_size
        self.search_table = dict()
        self.best_move = chess.Move.null()
        self.is_white = is_white
        self.board = board
        self.zobTable = [[[random.randint(1, 2 ** 64 - 1) for i in range(12)] for j in range(8)] for k in range(8)]
        self.time = 0
        self.time_limit = 0
        self.evaluator = BRViT(patch_size=4,
                               din=(1, 1, 8, 8),
                               dmodel=256,
                               dff=1024,
                               nheads=4,
                               nlayers=6,
                               dout=1,
                               out_activation=nn.Tanh(),
                               dropout=0.1)
        self.evaluator.load_state_dict(torch.load('./chessmodel'))
        self.evaluator.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.evaluator.to(self.device)

    def search(self, max_depth: int = 100, time_limit: float = 0.1):
        '''
        Method that starts a fail-soft Alpha-Beta search method using Negamax (computationally more efficient version
        of minimax) with transposition tables and iterative deeping that searches until the max_depth or time limit has
        been reached, whichever occurs first.

        Parameters:
            max_depth: 'int' - The maximum depth to explore during iterative deepening

            time_limit: 'float' - The amount of time in seconds to allow the engine to search
        '''
        # Initial time to calculate amount of time left
        self.time = time.time()
        self.time_limit = time_limit

        # Initial values for alpha, beta, and score
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        # Iterative Deepening
        for depth in range(max_depth):

            # Get children nodes and check if expanded
            move_order = list()
            for move in self.board.legal_moves:
                self.board.push(move)
                move_order.append((move, self.retrieve_table()))
                self.board.pop()

            # Sort by best score
            move_order = sorted(move_order, key=lambda x: (
            x[1] is not None, x[1] is not None and x[1]['node_type'] == 'PV',
            x[1] is not None and x[1]['node_type'] == 'Cut', x[1] is not None and x[1]['score']), reverse=True)

            # Traverse children of current node and return best score
            for move, node in move_order:
                self.board.push(move)
                board_value = -self.negamax(-beta, -alpha, 0, depth)

                # Check for best scoring move
                if board_value > best_score:
                    best_score = board_value
                    self.best_move = move
                if board_value > alpha:
                    alpha = board_value
                self.board.pop()

                # Check time after checking a move for a node
                if time.time() - self.time > self.time_limit:
                    for key in self.search_table.keys():
                        self.search_table[key]['age'] += 1
                    return

        # Update age for all positions in the table
        for key in self.search_table.keys():
            self.search_table[key]['age'] += 1

    def negamax(self, alpha: float, beta: float, curr_depth: int, max_depth: int):
        '''
        Negamax searching algorithm, a more computationally efficient variation of Minimax. When reaching final depth,
        performs a quiesce search to evaluate captures after the node to prevent bad trades. Utilizes fail-soft Alpha-
        Beta pruning.

        Parameters:
            alpha: 'float' - lower bound for cut-off range

            beta: 'float'  - upper bound for cut-off range

            curr_depth: 'int'   - Current depth

            max_depth: 'int'    - max depth to explore


        Returns -> ('float', 'list')  - Best score found so far for current node and a list of best moves representing
                                        the principal variation
        '''

        best_score = -math.inf
        best_move = chess.Move.null()

        if curr_depth == max_depth or self.board.is_game_over(claim_draw=True):
            return self.quiesce(alpha, beta)

        # Get children nodes and check if expanded
        move_order = list()
        for move in self.board.legal_moves:
            self.board.push(move)
            move_order.append((move, self.retrieve_table()))
            self.board.pop()

        # Sort by best score
        move_order = sorted(move_order, key=lambda x: (x[1] is not None, x[1] is not None and x[1]['node_type'] == 'PV',
                                                       x[1] is not None and x[1]['node_type'] == 'Cut',
                                                       x[1] is not None and x[1]['score']), reverse=True)

        for move, node in move_order:
            self.board.push(move)
            score = -self.negamax(-beta, -alpha, curr_depth + 1, max_depth)
            self.board.pop()

            if score >= beta:
                self.insert_table(best_score, best_move, curr_depth, 'Cut')
                return score
            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

            # Check time after checking a move for a node
            if time.time() - self.time > self.time_limit:
                self.insert_table(best_score, best_move, curr_depth, 'Cut')
                return best_score

        # All nodes have been explored so mark as either PV or All
        if best_score < alpha:  # Is an All node
            self.insert_table(best_score, best_move, curr_depth, 'All')
        else:
            self.insert_table(best_score, best_move, curr_depth, 'PV')

        return best_score

    def quiesce(self, alpha: float, beta: float):
        '''
        Quiesce search that does an additional search on potential captures to give a stable score for positions in which trading
        of pieces will occur to prevent bad trading that can result by not exploring captures further.

        Parameters:
            alpha: 'float' - lower bound for the cutoff score

            beta: 'float'  - upper bound for the cutoff score

        Returns -> 'float' - The evaluation score for the node
        '''

        stand_pat = self.evaluate()

        if not self.board.turn:
            stand_pat *= -1

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                self.board.push(move)
                score = -self.quiesce(-beta, -alpha)
                self.board.pop()

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

        return alpha

    def zobrist_hash(self):
        '''
        Zobrist hashing function, a hashing function used for chess positions, to convert a board position into a unique
        key. Converts the current board state in the internal board into a key for the search table.
        '''

        mapping = {
            -1: 7,
            -2: 8,
            -3: 9,
            -4: 10,
            -5: 11,
            -6: 12,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: 6
        }

        img = fen2img(self.board.fen()).numpy().squeeze()

        hsh = 0
        for i in range(8):
            for j in range(8):
                if img[i, j] != 0:
                    piece = mapping[img[i, j]]
                    hsh ^= self.zobTable[i][j][piece - 1]
        return int(hsh) % int(self.TABLE_LIMIT)

    def retrieve_table(self):
        '''
        Function to get position from the search table
        '''
        key = self.zobrist_hash()
        new_key = key
        while new_key in self.search_table.keys():
            if self.board.fen() == self.search_table[new_key]['fen']:
                self.search_table[new_key]['age'] = 0
                return self.search_table[new_key]

            new_key = (new_key + 1) % int(self.TABLE_LIMIT)

            if new_key == key:
                break

        return None

    def insert_table(self, score: float, move: chess.Move, depth: int, node_type: str):
        '''
        Function to insert a position into the search table storing

        Parameters:
            move: 'chess.Move' - the best move evaluated at that position
            score: 'float' - the best score evaluated at that position
            depth: 'int' - the depth of the search tree the position was explored
            node_type: 'str' - what type of node: PV, Cut, All
                                PV nodes are principal variation nodes - nodes in which the exact score is known, every child explored, and is the best node at its depth
                                Cut nodes are nodes that were not fully explored and triggered a beta cutoff
                                All nodes are nodes in which every child has been fully explored but has a score lower than the alpha in the search tree
        '''

        key = self.zobrist_hash()

        # Index is not in table
        if key not in self.search_table.keys():
            self.search_table[key] = {
                'fen': self.board.fen(),
                'best': move,
                'score': score,
                'depth': depth,
                'age': 0,
                'node_type': node_type
            }
            return

        # Index is in table
        new_key = key
        while new_key in self.search_table.keys():
            # Check if position in table
            if self.search_table[new_key]['fen'] == self.board.fen():
                if score > self.search_table[new_key]['score']:
                    self.search_table[new_key]['score'] = score
                    self.search_table[new_key]['move'] = move
                    self.search_table[new_key]['depth'] = depth
                self.search_table[new_key]['age'] = 0
                return

            new_key = (new_key + 1) % self.TABLE_LIMIT

            # Position not in table
            if new_key == key:
                break

        # Empty spot
        if new_key != key:
            self.search_table[key] = {
                'fen': self.board.fen(),
                'best': move,
                'score': score,
                'depth': depth,
                'age': 0,
                'node_type': node_type
            }
            return

        # Went through full table, replace entry
        entry = self.search_table[key]

        # Replace dependent upon age and depth searched
        entry_score = (0.5 * -entry['age']) + (0.5 * entry['depth'])
        new_score = 0.5 * depth

        if new_score > entry_score:
            self.search_table[key] = {
                'fen': self.board.fen(),
                'best': move,
                'score': score,
                'depth': depth,
                'age': 0,
                'node_type': node_type
            }


    def evaluate(self):
        '''
        Utility function to calculate an evaluation score for a given chess position

        Board is converted to a numpy array in which pawn, knight, bishop, rook, queen, and king are mapped to 1,2,3,4,5,6 respectively.
        Negative values represent black pieces, positive values represent white
        '''
        if self.board.is_checkmate():
            return -1 if self.board.turn else 1
        if self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        img, hidden = torch.unsqueeze(fen2img(self.board.fen()), dim=0), torch.zeros(1, 4, 256)
        img, hidden = img.to(self.device), hidden.to(self.device)
        score = self.evaluator(img, hidden)[0]
        return float(score.flatten()[0])

    def make_move(self, depth: int = 100, time_limit: float = 0.1):
        '''
        Function to search for a move from the current board state within the maximum depth or time limit (whichever occurs first)

        Parameters:
            depth: 'int' - The maximum depth to search to
            time_limit: 'float' - The maximum time in seconds to search for a move
        '''
        self.best_move = chess.Move.null()
        self.search(max_depth=depth, time_limit=time_limit)
        self.board.push(self.best_move)
        return self.best_move

    def opponent_move(self, move: chess.Move):
        '''
        Function required to be called after opponent's move to ensure internal board state matches current board state

        Parameters:
            move: 'chess.Move' - The move to push to the board

        '''
        self.board.push(move)


if __name__ == '__main__':
    # Perform test analysis
    results = list()

    # Create engines
    engine = chess.engine.SimpleEngine.popen_uci("C:/Users/Braden/Downloads/stockfish_15.1_win_x64_avx2/stockfish_15.1_win_x64_avx2/stockfish-windows-2022-x86-64-avx2.exe")
    knightingale = Knightingale(2 ** 30, False, chess.Board())

    # Play for 100 rounds alternating colors
    for round in range(100):
        # Track moves made in match
        movehistory = []

        # Create game object and set headers for file storage of game in pgn format
        game = chess.pgn.Game()
        game.headers['Date'] = datetime.now().strftime('%Y.%m.%d')
        game.headers['Round'] = str(round+1)
        game.headers['White'] = 'Stockfish 15' if round % 2 == 0 else 'Knightingale'
        game.headers['Black'] = 'Knightingale' if round % 2 == 0 else 'Stockfish 15'

        # Instantiate board
        board = chess.Board()

        # Start knightingale engine to default values and alternate colors every round
        knightingale.search_table.clear()
        knightingale.is_white = False if round % 2 == 0 else True
        knightingale.board = board.copy()

        # Play game until game is over
        while not board.is_game_over(claim_draw=True):
            # If the current color to move is not Knightingale then Stockfish plays
            if board.turn != knightingale.is_white:
                move = engine.play(board, chess.engine.Limit(time=0.5))
                movehistory.append(move.move)
                board.push(move.move)
                knightingale.opponent_move(move.move)
            else: # Otherwise Knightngale plays
                move = knightingale.make_move(time_limit=0.5)
                movehistory.append(move)
                board.push(move)

        # Add the mainline and results to the game object
        game.add_line(movehistory)
        game.headers["Result"] = str(board.result(claim_draw=True))

        # Print game object in pgn format to file
        print(game, file=open('./games.pgn', 'a+'), end='\n\n')

        # Determine winner or draw and print round results to console
        if eval(board.result()) == 0:
            winner = 'Draw'
        else:
            winner = "Stockfish 15" if board.turn == knightingale.is_white else "Knightingale"

        print(f'Round: {round+1}\tWinner: {winner}')

    # After all games close Stockfish
    engine.close()
