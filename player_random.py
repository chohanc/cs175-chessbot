import random

class RandomPlayer:
    def __init__(self):
        self.name = "random_player"
        
    def select_move(self, state, legal_moves):
        return random.choice(legal_moves)