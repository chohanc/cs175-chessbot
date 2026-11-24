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
from copy import deepcopy

NUM_NORMAL_ACTIONS = 64 * 64

PROMOTION_PATHS = (
    # White promotions
    "a7a8", "a7b8",
    "b7a8", "b7b8", "b7c8",
    "c7b8", "c7c8", "c7d8",
    "d7c8", "d7d8", "d7e8",
    "e7d8", "e7e8", "e7f8",
    "f7e8", "f7f8", "f7g8",
    "g7f8", "g7g8", "g7h8",
    "h7g8", "h7h8",

    # Black promotions
    "a2a1", "a2b1",
    "b2a1", "b2b1", "b2c1",
    "c2b1", "c2c1", "c2d1",
    "d2c1", "d2d1", "d2e1",
    "e2d1", "e2e1", "e2f1",
    "f2e1", "f2f1", "f2g1",
    "g2f1", "g2g1", "g2h1",
    "h2g1", "h2h1",
)

PROMOTION_MOVES = [
    path + piece
    for path in PROMOTION_PATHS
    for piece in ("q", "r", "b", "n")
]


PROMOTION_TO_ID = {
    move: action_id + NUM_NORMAL_ACTIONS
    for action_id, move in enumerate(PROMOTION_MOVES)
}

# reduced num_actions from 20,480 to 4272
NUM_ACTIONS = NUM_NORMAL_ACTIONS + len(PROMOTION_MOVES)  # 4272

def move_to_id(uci_move):
    """Encode a UCI move as a network output index."""
    move = chess.Move.from_uci(uci_move)

    if move.promotion is None:
        return move.from_square * 64 + move.to_square

    return PROMOTION_TO_ID[uci_move]

def id_to_move(action_id):
    """Decode an action ID to UCI. ChessEnv checks board legality."""
    if not 0 <= action_id < NUM_ACTIONS:
        raise ValueError(f"Move ID out of range: {action_id}")

    if action_id < NUM_NORMAL_ACTIONS:
        from_square, to_square = divmod(action_id, 64)
        return chess.Move(from_square, to_square).uci()

    return PROMOTION_MOVES[action_id - NUM_NORMAL_ACTIONS]


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
    def __init__(self, gamma=0.99, lr=1e-4, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995,
                 batch_size=32, memory_size=50_000, warmup=256,
                 target_update_every=250, device=None,
                 *, model=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if model is None:
            model = QNetwork()

        self.model = model.to(self.device)
        self.target_model = deepcopy(self.model)
        self.target_model.eval()

        model_name = type(model).__name__
        self.name = model_name

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # (Q(s,a) - target)^2 from the proposal

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup = warmup
        self.target_update_every = target_update_every
        self.memory = deque(maxlen=memory_size)
        self.training_steps = 0
        self.games_trained = 0

    def select_move(self, state, legal_moves, training=True):
        """Select a legal UCI move; exploration occurs only in training mode."""
        if not legal_moves:
            raise ValueError("No legal moves are available")

        if training and random.random() < self.epsilon:
            return random.choice(legal_moves)

        with torch.no_grad():
            x = encode_board(state).unsqueeze(0).to(self.device)
            q_values = self.model(x)[0]
            ids = [move_to_id(move) for move in legal_moves]
            best_position = q_values[ids].argmax().item()
            return legal_moves[best_position]

    def remember(self, state, action, reward, next_state, done):
        """Replay entry: (state, action, reward, next_state, done)."""
        self.memory.append((state, action, float(reward), next_state, bool(done)))

    def train_step(self):
        """Learn from one random replay batch; returns loss or None if warming up."""

        if not self.memory:
            return None

        # if the last move was end of a game (done) learn from it immediately with other batch of moves
        latest = self.memory[-1]
        done = latest[4]
        ready = len(self.memory) >= max(self.batch_size, self.warmup)

        if done:
            batch = [latest]

            if ready:
                indices = random.sample(range(len(self.memory) - 1), self.batch_size - 1)
                batch.extend(self.memory[i] for i in indices)

        elif ready:
            batch = random.sample(self.memory, self.batch_size)

        else:
            return None


        states = torch.stack([encode_board(t[0]) for t in batch]).to(self.device)
        action_ids = torch.tensor(
            [move_to_id(t[1]) for t in batch],
            dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t[2] for t in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([encode_board(t[3]) for t in batch]).to(self.device)

        # Predicted value of each action actually played.
        q_pred = self.model(states).gather(1, action_ids).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_model(next_states)
            opponent_best = torch.zeros(len(batch), device = self.device)

            for i, (_, _, _, next_fen, done) in enumerate(batch):
                if done:
                    continue  # Terminal target is simply reward.
                board = chess.Board(next_fen)
                legal_ids = [move_to_id(m.uci()) for m in board.legal_moves]
                if legal_ids:
                    opponent_best[i] = next_q_all[i, legal_ids].max()

            # Zero-sum turn alternation: next Q-value belongs to the opponent.
            targets = rewards - self.gamma * opponent_best

        loss = self.loss_fn(q_pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return float(loss.item())

    def finish_episode(self):
        """Reduce randomness slowly after each self-play game."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.games_trained += 1

    def save(self, path="chess_dqn.pt"):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "target_model": self.target_model.state_dict(),
            "replay_memory": list(self.memory),
            "games_trained": self.games_trained,
        }, path)

    def load(self, path="chess_dqn.pt"):
        # Load only files you trust. The checkpoint should be one you saved.
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        #self.target_model.load_state_dict(self.model.state_dict())
        target_state = checkpoint.get("target_model", checkpoint["model"])
        self.target_model.load_state_dict(target_state)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.memory.clear()
        self.memory.extend(checkpoint.get("replay_memory", []))
        self.games_trained = checkpoint.get("games_trained", 0)
