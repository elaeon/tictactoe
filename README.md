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
python tictactoe.py --run-best-plays 100
```
Note: Maybe you will need to run this command
until you will get the four patterns that are showed with --game-pattern
