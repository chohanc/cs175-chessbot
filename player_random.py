import random

class RandomPlayer:
    def select_move(self, state, legal_moves):
        return random.choice(legal_moves)