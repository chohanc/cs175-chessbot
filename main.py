from game_runner import GameRunner
from player_random import RandomPlayer
from player_stockfish import StockfishPlayer
from player import SelfLearningPlayer

def main(player1, player2, player1_color, record_game):
    game = GameRunner(player1, player2, player1_color, record_game=True)
    game.run_game()
    game.print_results(with_move_history=True, with_board_history=True)
    game.delete_game_file()


if __name__ == "__main__":
    player1 = RandomPlayer()
    player2 = StockfishPlayer("/opt/homebrew/bin/stockfish")
    try:
        main(player1, player2, "white", False)
    finally:
        player2.close()