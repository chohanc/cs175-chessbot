import csv
from game_runner import GameRunner

"""
Eval will allow for testing of multiple games of a single and be
responsible for recording all of the stats (going to finish this part thursday)
"""

class Eval:
    def __init__(self, player1, player2):
        self.bot = player1
        self.opponent = player2

    def run_games(self, num_games):
        if num_games < 1:
            raise ValueError("num_games must be at least 1")

        wins = draws = losses = 0
        total_seconds = 0.0
        total_plies = 0

        for i in range(num_games):
            bot_color = "white" if i % 2 == 0 else "black"
            game = GameRunner(self.bot, self.opponent, bot_color)
            result = game.run_game()

            winner_color = game.env.get_winner()
            if winner_color is None:
                draws += 1
                outcome = "DRAW"
            elif winner_color == bot_color:
                wins += 1
                outcome = "WIN"
            else:
                losses += 1
                outcome = "LOSS"

            played = i + 1
            print(
                f"[Eval {played}/{num_games}] {bot_color} {outcome} | "
                f"{result['num_plies']} plies | {result['duration']:.2f}s | "
                f"W/D/L {wins}/{draws}/{losses} | win rate {wins / played:.1%}",
                flush=True,
            )

            total_seconds += result["duration"]
            total_plies += result["num_plies"]

        stats = {
            "bot": self.bot.name,
            "opponent": self.opponent.name,
            "games": num_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate_pct": round(100 * wins / num_games, 2),
            "avg_game_seconds": round(total_seconds / num_games, 2),
            "avg_plies": round(total_plies / num_games, 2),
        }
        self.record_stats(stats)
        return stats

    def record_stats(self, stats):
        with open("eval_summary.csv", "a", newline = "") as file:
            writer = csv.DictWriter(file, fieldnames = stats.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(stats)