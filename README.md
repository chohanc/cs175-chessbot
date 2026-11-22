# cs175-chessbot
Run   
  
**pip install -r .\requirement.txt**  
and  
**python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu**  
  
to install libraries

## KQ vs K mate-in-one experiment

Run from this directory on macOS with Python 3.12. The experiment uses one CPU
thread. The reference run used Python 3.12.0 on Apple silicon.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install chess==1.11.2 torch==2.8.0 numpy==2.2.5
.venv/bin/python mate_experiment.py --checkpoint mate_dqn.pt > rerun.json
diff -u mate_results.json rerun.json
```

The script samples legal, White-to-move king-and-queen vs. king positions and
retains boards with at least one immediate mate. Split seed 175 fixes the first
eight distinct positions for practice and the next 32 for evaluation. Model and
training seed 24 fixes the initial weights, exploration, replay sampling and
practice order.

Each of 300 rounds visits every practice position once in shuffled order. The
existing DQN chooses a legal move and receives the environment reward, +1 for
checkmate and 0 for a miss. Every attempt ends after that move. Training uses
the `alex` replay and optimizer defaults with epsilon decay set to 0.999 per
attempt. Evaluation calls `select_move(..., training=False)` once per board
before the first update and after the final update. Any legal checkmate counts.
The JSON records both selected moves in UCI notation for every FEN.

| Split | Before training | After 2,145 updates |
| --- | ---: | ---: |
| Practice | 1/8 (12.5%) | 8/8 (100%) |
| Held out | 3/32 (9.375%) | 2/32 (6.25%) |

On the held-out set, training gained one solve and lost two. The model learned
the practice positions, but this single run showed no improvement on unseen
positions. See [the move-by-move results](mate_results.json).

