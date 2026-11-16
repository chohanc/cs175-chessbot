
import chess


class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    # Start a new game
    def reset(self):
        self.board.reset()
        return self.get_state()

    # Return the current board state as a FEN string
    def get_state(self):
        return self.board.fen()

    # Return all legal moves as UCI strings
    def get_legal_moves(self):
        if self.is_game_over():
            return []

        return [move.uci() for move in self.board.legal_moves]

    # Return whose turn it is
    def get_turn(self):
        return "white" if self.board.turn == chess.WHITE else "black"

    # Check whether the game has ended
    def is_game_over(self):
        return self.board.is_game_over(claim_draw=True)

    # Return the final result (1-0, 0-1, 1/2-1/2, or *)
    def get_result(self):
        return self.board.result(claim_draw=True)

    # Return the winner, or None if drawn or ongoing
    def get_winner(self):
        outcome = self.board.outcome(claim_draw=True)

        if outcome is None or outcome.winner is None:
            return None

        return "white" if outcome.winner == chess.WHITE else "black"

    # Execute a move and return the updated game information
    def step(self, action):
        if self.is_game_over():
            raise ValueError("Game is already over")

        # Convert UCI string to a chess.Move object
        move = chess.Move.from_uci(action)

        # Validate the move before executing it
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {action}")

        # Remember which player is making the move
        player = self.board.turn

        # Execute the move
        self.board.push(move)

        # Check the outcome after the move
        outcome = self.board.outcome(claim_draw=True)
        done = outcome is not None

        # Reward from the perspective of the player who moved
        reward = 0.0

        if done and outcome.winner is not None:
            reward = 1.0 if outcome.winner == player else -1.0

        return self.get_state(), reward, done

    # Display the current chessboard
    def render(self):
        print(self.board)
