import chess
from chess_env import ChessEnv
import time
import csv
import os

class GameRunner:
    def __init__(self, player1, player2, player1_color = "white", *, record_game = False):
        self.env = ChessEnv()
        self.player1 = player1
        self.player2 = player2
        self.record_game = record_game
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
        # collects all info for a single game
        # adds initial board position
        self.game_result["board_history"].append(self.env.get_fen())
        start_time = time.time()
        done = False
        while not done:
            turn = self.env.get_turn()
            if turn == "white":
                player = self.white_player
            else:
                player = self.black_player

            action = player.select_move(
                state,
                self.env.get_legal_moves()
            )

            state, _, done = self.env.step(action)

            self.game_result["num_plies"] += 1
            self.game_result["move_history"].append({"player": turn,
                                                "move": action})
            self.game_result["board_history"].append(self.env.get_fen())
        
        self.game_result["duration"] = time.time() - start_time
        self.game_result["winner"]= "player 1" if (self.env.get_result() == "1-0") else "player 2"
        self.game_result["termination"] = self.env.get_termination()
        if self.record_game is True:
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


