import chess
import time
import csv
import os
from chess_env import ChessEnv
from player import SelfLearningPlayer
from player_NNandABprune import SelfLearningPlayer as HybridPlayer

class GameRunner:
    def __init__(self, player1, player2, player1_color="white", *,
                 record_game=False, player1_depth=0, player2_depth=0):
        self.env = ChessEnv()
        self.player1 = player1
        self.player2 = player2
        self.record_game = record_game
        self.player1_depth = player1_depth
        self.player2_depth = player2_depth
        self.white_player = None
        self.black_player = None
        self.assign_color(player1_color)

        self.game_result = {
            "white_player": self.white_player.name,
            "black_player": self.black_player.name,
            "winner": None,
            "termination": None,
            "num_plies": 0,
            "move_history": [],
            "board_history": []
        }
    

    def assign_color(self, player1_color):
        self.player1_color = player1_color
        if player1_color == "white":
            self.white_player = self.player1
            self.black_player = self.player2
        else:
            self.white_player = self.player2
            self.black_player = self.player1


    def run_game(self):
        state = self.env.reset()
        self.game_result["board_history"].append(self.env.get_fen())
        start_time = time.time()
        done = False

        while not done:
            turn = self.env.get_turn()
            if turn == "white":
                player = self.white_player
            else:
                player = self.black_player

            legal_moves = self.env.get_legal_moves()
            depth = self.player1_depth if player is self.player1 else self.player2_depth

            move_start = time.perf_counter()

            if isinstance(player, HybridPlayer):
                action = player.select_move(
                    state, legal_moves,
                    training=False,
                    search_depth=depth
                )
            elif isinstance(player, SelfLearningPlayer):
                action = player.select_move(
                    state, legal_moves,
                    training=False
                )
            else:
                action = player.select_move(state, legal_moves)

            elapsed = time.perf_counter() - move_start

            state, _, done = self.env.step(action)

            self.game_result["num_plies"] += 1
            self.game_result["move_history"].append({
                "player": turn,
                "move": action
            })
            self.game_result["board_history"].append(self.env.get_fen())

            print(f"Ply {self.game_result['num_plies']} | "
                  f"{turn} | {action} | {elapsed:.3f}s", flush=True)

        self.game_result["duration"] = time.time() - start_time

        result = self.env.get_result()
        if result == "1-0":
            winner = self.white_player
        elif result == "0-1":
            winner = self.black_player
        else:
            winner = None

        if winner is None:
            self.game_result["winner"] = "draw"
        elif winner is self.player1:
            self.game_result["winner"] = "player 1"
        else:
            self.game_result["winner"] = "player 2"

        self.game_result["termination"] = self.env.get_termination()

        if self.record_game:
            self.save_game()

        return self.game_result
    

    def save_game(self, game_number=1, append=False):
        fieldnames = [
                      "game",
                      "white_player",
                      "black_player",
                      "winner",
                      "termination", 
                      "num_plies", 
                      "duration"]
        
        if append and game_number > 1:
            with open('game_results.csv', 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                writer.writerow({"game": game_number} | self.game_result)
        else:
            with open('game_results.csv', 'w',newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerow({"game": game_number} | self.game_result)


    def delete_game_file(self):
        if os.path.exists("game_results.csv"):
            os.remove("game_results.csv")
        else:
            print("game_results.csv not found")


    def print_results(self, *, with_move_history=False, with_board_history=False):
        if with_move_history or with_board_history:
            if with_board_history:
                print(chess.Board(self.game_result["board_history"][0]))
                
            for x in range(1, self.game_result["num_plies"] + 1):
                print(f"move {x}: ", self.game_result["move_history"][x - 1])
                if with_board_history:
                        print(chess.Board(self.game_result["board_history"][x]))
        print("winner:", self.game_result["winner"])
        print("termination:", self.game_result["termination"])        


