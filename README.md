# tic-tac-toe
Script to find the bests patterns in the game.

To show the best paterns run:
```
python tictactoe.py --game-pattern
```

### To recreate the results saved in game_patterns.txt do the next steps:

This command generates all permutations in the game. 
Also, duplicated moves and symmetric moves are removed from the final list 
which contains the best 6 patterns for X and O.
```
python tictactoe.py --generate-all-moves
```

In this step you can run a fixed number of n iterations (per default 100) of random games using 
the best 6 patterns found in the previous step, and the results are saved in games_patterns.txt
```
python tictactoe.py --run-best-plays 200
```
Note: If you get an error running --game-pattern again, then just increase a little bit the number of iterations.
