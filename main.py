from player_random import RandomPlayer
from player_stockfish import StockfishPlayer
from player_NNandABprune import SelfLearningPlayer, CNNQNetwork
from game_runner import GameRunner

def main(player1, player2, player1_color, record_game,
         player1_depth=0, player2_depth=0):
    game = GameRunner(
            player1, player2, player1_color,
            record_game=record_game,
            player1_depth=player1_depth,
            player2_depth=player2_depth)
    game.run_game()
    game.print_results(
        with_move_history=False,
        with_board_history=False
    )

if __name__ == "__main__":
    player1 = SelfLearningPlayer(model=CNNQNetwork())
    player1.load("cnn_c32_b3_lr0.0001_bs32.pt")

    player2 = RandomPlayer()

    main(player1, player2, "white", True,
         player1_depth=2)