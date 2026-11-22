"""Self-play training with a shared DQN for White and Black.

Requires your existing chess_env.py implementing ChessEnv.reset(),
get_state(), get_legal_moves(), step(), is_game_over(), get_result().
"""
from collections import Counter

from chess_env import ChessEnv
from player import SelfLearningPlayer


NUM_GAMES = 10              # Quick starting run; increase for longer training.
MAX_PLIES = 300             # Safety cap: a capped game gets 0 terminal reward.
SAVE_EVERY = 5


def main():
    env = ChessEnv()
    player = SelfLearningPlayer()
    results = Counter()

    for game in range(1, NUM_GAMES + 1):
        state = env.reset()
        recent_losses = []
        result = "capped"

        for ply in range(MAX_PLIES):
            legal_moves = env.get_legal_moves()
            action = player.select_move(state, legal_moves, training=True)
            next_state, reward, done = env.step(action)

            # End exceptionally long games for training; treat cap as a draw.
            capped = (ply + 1 == MAX_PLIES) and not done
            player.remember(state, action, reward if not capped else 0.0,
                            next_state, done or capped)
            loss = player.train_step()
            if loss is not None:
                recent_losses.append(loss)
            state = next_state

            if done:
                result = env.get_result()
                break
            if capped:
                break

        player.finish_episode()
        results[result] += 1

        if game % 10 == 0:
            mean_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
            loss_display = f"{mean_loss:.5f}" if mean_loss is not None else "warming up"
            print(f"Game {game}/{NUM_GAMES} | last={result} | "
                  f"epsilon={player.epsilon:.3f} | loss={loss_display} | "
                  f"results={dict(results)}")

        if game % SAVE_EVERY == 0:
            player.save("chess_dqn.pt")

    player.save("chess_dqn.pt")
    print("Saved trained network to chess_dqn.pt")


if __name__ == "__main__":
    main()
