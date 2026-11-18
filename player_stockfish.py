import chess
import chess.engine
"""
Initializes Stockfish Chess engine and retrieves moves to play
"""

class StockfishPlayer:
    def __init__(self, stockfish_path, elo=1500, time_limit=0.1, threads=1, hash_size=32):
        self.name = "stockfish"
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo,
            "Threads": threads,
            "Hash": hash_size
        })
        self.time_limit = time_limit

    def select_move(self, state, legal_moves):
        board = chess.Board(state)
        result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        
        return result.move.uci()
        
    def close(self):
        self.engine.quit()