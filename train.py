"""Train a shared DQN chess player through self-play."""

from collections import Counter
from pathlib import Path

from chess_env import ChessEnv
from player import CNNQNetwork, QNetwork, SelfLearningPlayer


NUM_GAMES = 10
MAX_PLIES = 1000
SAVE_EVERY = 5

CNN_CHANNELS = 32
CNN_BLOCKS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 32


def main():
    choice = input("Choose a model [cnn/mlp, default: cnn]: ").strip().lower()
    num_games = int(input(f"Number of training games [default: {NUM_GAMES}]: ").strip() or NUM_GAMES)

    if choice in ("", "cnn"):
        model = CNNQNetwork(channels=CNN_CHANNELS, num_blocks=CNN_BLOCKS)
        model_name = f"cnn_c{CNN_CHANNELS}_b{CNN_BLOCKS}"
    elif choice == "mlp":
        model = QNetwork()
        model_name = "mlp"
    else:
        print("Please enter 'cnn' or 'mlp'.")
        return

    default_name = f"{model_name}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.pt"
    name = input(f"Checkpoint filename [default: {default_name}]: ").strip()
    filename = Path(name or default_name)

    player = SelfLearningPlayer(
        model=model,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )

    if filename.is_file() and filename.stat().st_size > 0:
        player.load(filename)
        # Use the learning rate selected above when continuing training.
        player.optimizer.param_groups[0]["lr"] = LEARNING_RATE
        print(f"Resuming training from {filename}")
    else:
        print(f"Starting a new training run. Checkpoint: {filename}")

    env = ChessEnv()
    results = Counter()

    print(f"Model: {player.name}")
    print(f"Device: {player.device}")
    print(f"Training games: {num_games}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")

    # NEW: Display previous training progress
    print(f"Previously trained games: {player.games_trained}")
    print(f"Previous training steps: {player.training_steps}")

    for game in range(1, num_games + 1):
        state = env.reset()
        losses = []

        for ply in range(1, MAX_PLIES + 1):
            legal_moves = env.get_legal_moves()
            action = player.select_move(state, legal_moves, training=True)
            next_state, reward, done = env.step(action)

            capped = ply == MAX_PLIES and not done
            if capped:
                reward = 0.0

            player.remember(state, action, reward, next_state, done or capped)
            loss = player.train_step()
            if loss is not None:
                losses.append(loss)

            if done or capped:
                result = env.get_result() if done else "capped"
                break

            state = next_state

        player.finish_episode()
        results[result] += 1

        loss_text = f"{sum(losses) / len(losses):.5f}" if losses else "warming up"
        print(
            f"Game {game}/{num_games} | "
            f"total_games={player.games_trained} | "  # NEW
            f"result={result} | plies={ply} | "
            f"epsilon={player.epsilon:.3f} | loss={loss_text} | "
            f"results={dict(results)}"
        )

        if game % SAVE_EVERY == 0 or game == num_games:
            player.save(filename)

    # NEW: Display cumulative training progress
    print(f"Total games trained: {player.games_trained}")
    print(f"Total training steps: {player.training_steps}")
    print(f"Saved trained network to {filename}")


if __name__ == "__main__":
    main()