"""Self-playing DQN chess agent compatible with ChessEnv

Inputs: FEN chess position and a list of legal moves as UCI strings.
Outputs: one legal move as a UCI string.
Learning: shared-network, zero-sum Q-learning, replay, target network.
"""

from collections import deque
import random

import chess
import torch
from torch import nn


NUM_ACTIONS = 5 * 64 * 64  # 20,480
# Codes: 0 = no promotion, 1 = queen, 2 = rook, 3 = bishop, 4 = knight
PROMOTION_CODE = {None: 0, chess.QUEEN: 1, chess.ROOK: 2,
                  chess.BISHOP: 3, chess.KNIGHT: 4}
CODE_PROMOTION = {code: piece for piece, code in PROMOTION_CODE.items()}


def move_to_id(uci_move):
    """Consistent with the project proposal's 5 * 64 * 64 action scheme."""
    move = chess.Move.from_uci(uci_move)
    code = PROMOTION_CODE[move.promotion]
    return code * 4096 + move.from_square * 64 + move.to_square


def id_to_move(action_id):
    """Decode an action ID to UCI; legality must be checked by ChessEnv."""
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError(f"Move Id out of range: {action_id}")
    code, square_pair = divmod(action_id, 4096)
    from_square, to_square = divmod(square_pair, 64)
    return chess.Move(from_square, to_square, promotion=CODE_PROMOTION[code]).uci()


def encode_board(fen):
    """18 x 8 x 8: 12 piece planes, turn, four castling rights, en passant."""
    board = chess.Board(fen)
    x = torch.zeros((18, 8, 8), dtype=torch.float32)

    # Piece planes 0..5 white P,N,B,R,Q,K; 6..11 black P,N,B,R,Q,K.
    for square, piece in board.piece_map().items():
        channel = (0 if piece.color == chess.WHITE else 6) + piece.piece_type - 1
        row = 7 - chess.square_rank(square)  # row 0 is rank 8
        col = chess.square_file(square)
        x[channel, row, col] = 1.0

    x[12].fill_(float(board.turn == chess.WHITE))
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    for channel, available in enumerate(rights, start=13):
        x[channel].fill_(float(available))

    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        x[17, row, col] = 1.0
    return x


class ResidualBlock(nn.Module):
    """Refine board features while preserving the input through a skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.relu(self.conv1(x))
        features = self.conv2(features)
        return self.relu(x + features)


class CNNQNetwork(nn.Module):
    """Estimate one Q-value per action from an 18-channel chessboard."""
    # we can tune channels and num_blocks across gameplay
    def __init__(self, channels: int = 32, num_blocks: int = 3):
        super().__init__()

        # Keep one feature vector at each of the 64 board squares.
        self.input_layer = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_blocks))
        )

        # Preserve square locations when mapping board features to action IDs.
        self.q_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(states)
        x = self.blocks(x)
        return self.q_head(x)


class QNetwork(nn.Module):
    """Small MLP producing one Q-value per possible action."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )

    def forward(self, states):
        return self.layers(states)

class SelfLearningPlayer:
    def __init__(self):
        return

