from chess_env import ChessEnv
from player_random import RandomPlayer
from player import SelfLearningPlayer

def main():
    env = ChessEnv()

    white_player = RandomPlayer()
    black_player = RandomPlayer()

    state = env.reset()

    # "w" overwrites the previous game
    with open("latest_game.txt", "w", encoding="utf-8") as file:

        file.write("=== NEW CHESS GAME ===\n\n")

        move_number = 0

        while not env.is_game_over():
            if env.get_turn() == "white":
                player = white_player
            else:
                player = black_player

            action = player.select_move(
                state,
                env.get_legal_moves()
            )

            state, reward, done = env.step(action)

            move_number += 1

            # Record the move
            file.write(f"Move {move_number}: {action}\n")

            # Get the current board state
            board_state = str(env.show_board())

            file.write(board_state + "\n\n")

            print(f"Move {move_number}: {action}")
            print()

        result = env.get_result()

        file.write(f"Game result: {result}\n")

    print("Game result:", result)
    print("Game saved to latest_game.txt")



if __name__ == "__main__":
    main()