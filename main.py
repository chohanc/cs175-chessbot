from chess_env import ChessEnv
from player_random import RandomPlayer
from player_stockfish import StockfishPlayer
from player import SelfLearningPlayer
import time

def main(player1, player2):
    env = ChessEnv()
    white_player = player1
    black_player = player2

    state = env.reset()
    # collects all info for a single game
    game_result = {
        "white_player": white_player.name,
        "black_player": black_player.name,
        "winner": None,
        "termination": None,
        "num_plies": 0,
        "move_history": [],
        "board_history": []
    }
    # adds initial board position
    game_result["board_history"].append(env.get_fen())
    start_time = time.time()
    done = False
    while not done:
        turn = env.get_turn()
        if turn == "white":
            player = white_player
        else:
            player = black_player

        action = player.select_move(
            state,
            env.get_legal_moves()
        )

        state, reward, done = env.step(action)

        game_result["num_plies"] += 1
        game_result["move_history"].append({"player": turn,
                                            "move": action})
        game_result["board_history"].append(env.get_fen())


    game_result["duration"] = time.time() - start_time
    game_result["winner"]= env.get_result()
    game_result["termination"] = env.get_termination()
        

    print("Game result:", game_result["winner"])
    print("moves_made:", game_result["move_history"])
    print(len(game_result["move_history"]) == game_result["num_plies"])

    print(len(game_result["board_history"]) == game_result["num_plies"] + 1)
    print(env.get_result())
    print(env.get_termination())
    return game_result



if __name__ == "__main__":
    player1 = RandomPlayer()
    player2 = StockfishPlayer("/opt/homebrew/bin/stockfish")

    try:
        main(player1, player2)
    finally:
        player2.close()