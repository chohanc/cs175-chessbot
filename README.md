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
.venv/bin/python mate_experiment.py --practice-positions 64 --checkpoint mate_dqn_64.pt > rerun_64.json
diff -u mate_results_64.json rerun_64.json
```

The script samples legal, White-to-move king-and-queen vs. king positions and
retains boards with at least one immediate mate. Split seed 175 fixes the first
eight distinct positions for practice and the next 32 for evaluation. The
64-position run adds the next 56 positions to the original eight. Model and
training seed 24 fixes the initial weights, exploration, replay sampling and
practice order for each run.

Each run makes 2,400 training attempts in shuffled rounds. The eight-position
run visits each board 300 times. The 64-position run visits each board 37 times,
then samples 32 boards for one final attempt each. The existing DQN chooses a
legal move and receives the environment reward, +1 for checkmate and 0 for a
miss. Every attempt ends after that move. Training uses
the `alex` replay and optimizer defaults with epsilon decay set to 0.999 per
attempt. Evaluation calls `select_move(..., training=False)` once per board
before the first update and after the final update. Any legal checkmate counts.
The JSON records both selected moves in UCI notation for every FEN.

| Practice boards | Split | Before training | After 2,145 updates |
| ---: | --- | ---: | ---: |
| 8 | Practice | 1/8 (12.5%) | 8/8 (100%) |
| 8 | Held out | 3/32 (9.375%) | 2/32 (6.25%) |
| 64 | Practice | 3/64 (4.69%) | 51/64 (79.69%) |
| 64 | Held out | 3/32 (9.375%) | 4/32 (12.5%) |

The 64-position run solved two more held-out boards than the eight-position run
and one more than their shared starting model. Both runs used the same held-out
boards and a single initialization seed. See the per-position results for
[8 practice boards](mate_results.json) and [64 practice boards](mate_results_64.json).

