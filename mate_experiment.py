"""One-ply KQ vs K experiment using the alex branch's DQN and replay trainer."""

import argparse
import json
import platform
import random

import chess
import torch

from chess_env import ChessEnv
from player import SelfLearningPlayer


def make_positions(count, seed):
    """Sample distinct legal, White-to-move KQK boards admitting immediate mate."""
    rng = random.Random(seed)
    positions = []
    while len(positions) < count:
        board = chess.Board.empty()
        for square, piece in zip(rng.sample(range(64), 3), "KQk"):
            board.set_piece_at(square, chess.Piece.from_symbol(piece))
        if not board.is_valid() or board.fen() in positions:
            continue
        for move in list(board.legal_moves):
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                positions.append(board.fen())
                break
    return positions


def evaluate(player, positions):
    records = {}
    for fen in positions:
        board = chess.Board(fen)
        legal = [m.uci() for m in board.legal_moves]
        action = player.select_move(fen, legal, training=False)
        board.push_uci(action)
        records[fen] = {"move": action, "mate": board.is_checkmate()}
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="optional path for the trained DQN")
    args = parser.parse_args()
    model_seed, split_seed = 24, 175
    rounds = 300
    positions = make_positions(40, split_seed)
    practice, held_out = positions[:8], positions[8:]
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    player = SelfLearningPlayer(device="cpu", epsilon_decay=0.999)
    env = ChessEnv()
    before = evaluate(player, positions)

    for _ in range(rounds):
        for fen in random.sample(practice, len(practice)):
            env.board.set_fen(fen)
            action = player.select_move(fen, env.get_legal_moves(), training=True)
            next_state, reward, _ = env.step(action)
            # Every puzzle ends after one move, including misses.
            player.remember(fen, action, reward, next_state, done=True)
            player.train_step()
            player.finish_episode()

    after = evaluate(player, positions)
    if args.checkpoint:
        player.save(args.checkpoint)
    report = {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "chess": chess.__version__, "device": str(player.device),
        "model_and_training_seed": model_seed, "split_seed": split_seed,
        "attempts_per_practice_position": rounds, "updates": player.training_steps,
    }
    for name, split in [("practice", practice), ("held_out", held_out)]:
        report[name] = {
            "total": len(split),
            "solved_before": sum(before[fen]["mate"] for fen in split),
            "solved_after": sum(after[fen]["mate"] for fen in split),
            "positions": [{"fen": fen, "before": before[fen], "after": after[fen]}
                          for fen in split],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
