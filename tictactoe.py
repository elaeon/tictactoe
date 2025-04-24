import numpy as np
from itertools import permutations, product
from dataclasses import dataclass
import heapq
from collections import defaultdict
from random import choice
# Import needed only for the side-by-side example
from importlib import import_module
import argparse
import configparser


@dataclass
class Player:
    X = -1
    O = 1
    U = 0

    @classmethod
    def to_str(cls, v):
        if cls.X == v:
            return "X"
        elif cls.O == v:
            return "O"
        else:
            return " "


class Board:
    slice_positions = {
        0: (slice(0, 1), slice(0, 3)),
        1: (slice(1, 2), slice(0, 3)),
        2: (slice(2, 3), slice(0, 3)),
        3: (slice(0, 3), slice(0, 1)),
        4: (slice(0, 3), slice(1, 2)),
        5: (slice(0, 3), slice(2, 3))
    }

    def __init__(self, grid_width=3, grid_height=3):
        self.grid = self.set_grid()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.moves = []

    def winner(self) -> int:
        winner, _, _ = self.check_board()
        if winner == -3:
            return Player.X
        elif winner == 3:
            return Player.O
        elif winner == 0:
            return Player.U
        else:
            return None
    
    def check_board(self, status: str="win"):
        if status == "win":
            status_values = [-3, 3]
        else:
            status_values = [-2, 2]

        row_0 = self.grid[self.slice_positions[0]].reshape(-1)
        row_1 = self.grid[self.slice_positions[1]].reshape(-1)
        row_2 = self.grid[self.slice_positions[2]].reshape(-1)
        col_0 = self.grid[self.slice_positions[3]].reshape(-1)
        col_1 = self.grid[self.slice_positions[4]].reshape(-1)
        col_2 = self.grid[self.slice_positions[5]].reshape(-1)
        dig_0 = self.grid.diagonal()
        dig_1 = np.diag(np.fliplr(self.grid))
        for name_position, vector in [("r0", row_0), ("r1", row_1), ("r2", row_2), ("c0", col_0), ("c1", col_1), ("c2", col_2), ("d0", dig_0), ("d1", dig_1)]:
            winner = sum(vector)
            if winner in status_values:
                return winner, vector, name_position
        return 0, -1, ""
        

    def add_move(self, x, y, player):
        if self.grid[x, y] == 0:
            self.grid[x, y] = player
            return True
        else:
            return False


    def set_grid(self):
        grid = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        return grid

    def reset_game(self):
        self.set_grid()
     
    def print(self):
        print(",".join(self.moves))
        print("+" + "-" * (self.grid_width * 2 - 1) + "+")
        for row in self.grid:
            print("|" + "|".join([Player.to_str(e) for e in row]) + "|")
            print("+" + "-" * (self.grid_width * 2 - 1) + "+")


def get_grid_lines(board: Board):
    temp_stdout = import_module('io').StringIO()
    import sys
    old_stdout = sys.stdout
    sys.stdout = temp_stdout
    board.print()
    sys.stdout = old_stdout
    return temp_stdout.getvalue().splitlines()


def save_play_positions(positions_played):
    with open(positions_filename, "a") as f:
        for line in positions_played:
            f.write(",".join(map(str, line)))
            f.write("\n")


def save_metadata(player_X_winner, player_O_winner, player_draw):
    games = player_X_winner + player_O_winner + player_draw
    print(f"Win X: {player_X_winner}. Win O: {player_O_winner}. Draw: {player_draw}")
    print(f"Win X: {player_X_winner/games}. Win O: {player_O_winner/games}. Draw: {player_draw/games}")
    with open(positions_sts_filename, "w") as f:
        f.write(f"X: {Player.X}, O: {Player.O}, : {Player.U}")
        f.write("\n")
        f.write(f"{Player.X}: {player_X_winner}")
        f.write("\n")
        f.write(f"{Player.O}: {player_O_winner}")
        f.write("\n")
        f.write(f"{Player.U}: {player_draw}")
        f.write("\n")
        f.write(f"Games: {games}")
        f.write("\n")


def clean_play_positions(filename):
    with open(filename, "w") as f:
        f.write("")


def remove_duplicated_moves(input_filename, output_filename):
    with open(input_filename, "r") as f:
        table_indexes = set([])
        line = "-"
        while line != "":
            line = f.readline()
            table_indexes.add(line)

    with open(output_filename, "w") as f:
        for index in table_indexes:
            f.write(index)


def remove_symmetric_moves(input_filename, output_filename):
    with open(input_filename, "r") as f:
        table_indexes = set([])
        while True:
            moves_str: str = f.readline()
            if len(moves_str) > 0:
                moves = moves_str.split(",")
                moves_no_label = moves[:-1]
                rotation = rotate(moves_no_label)
                rotated_moves = rotate_moves(rotation, moves_no_label)
                reflected_moves = reflection_moves(rotated_moves)
                table_indexes.add(reflected_moves+","+moves[-1])
            else:
                break

    with open(output_filename, "w") as f:
        for index in table_indexes:
            f.write(index)


def save_win_or_lose_moves(table: dict, input_filename: str, output_filename: str):
    with open(input_filename, "r") as f:
        table_moves: set = set([])
        max_moves: int = 3
        while True:
            moves_str = f.readline()
            if len(moves_str) > 0:
                moves = moves_str.split(",")
                base_moves = moves[:max_moves]
                sts = get_sts(table, Player.O, base_moves)
                if sts["draw"] == 0:
                    base_moves_target = best_play_game(table, {}, moves=base_moves, print_board_on_terminal=False)
                    table_moves.add(",".join(base_moves_target[:3] + [base_moves_target[-1]]))
            else:
                break
                
    with open(output_filename, "w") as f:
        for move in sorted(table_moves):
            f.write(move)
            f.write("\n")


def play():
    player_X_winner = 0
    player_O_winner = 0
    player_draw = 0
    positions_played_batch = []
    batch_size = 100
    
    for positions_to_play in permutations(product([0,1,2], repeat=2), 9):
        board = Board()
        positions_played = []
        for player_turn, position in enumerate(positions_to_play):
            i, j = position
            active_player = Player.X if player_turn % 2 == 0 else Player.O
            if board.add_move(i, j, active_player) is True:
                winner_player = board.winner()
                positions_played.append(f"{i}{j}")
                if winner_player != Player.U:
                    if winner_player == Player.X:
                        player_X_winner += 1
                    else:
                        player_O_winner += 1
                    positions_played.append(winner_player)
                    break
            else:
                print(f"INVALID POSITION, {i, j}")
                return
        else:
            positions_played.append(winner_player)
            player_draw += 1
        positions_played_batch.append(positions_played)
        if len(positions_played_batch) == batch_size:
            save_play_positions(positions_played_batch)
            positions_played_batch = []
    
    save_play_positions(positions_played_batch)
    save_metadata(player_X_winner, player_O_winner, player_draw)


def moves_to_board(moves: list) -> Board:
    board = Board()
    board.moves = moves
    for player_turn, move in enumerate(moves):
        i = int(move[0])
        j = int(move[1])
        active_player = Player.X if player_turn % 2 == 0 else Player.O
        board.add_move(i, j, active_player)
    return board


def moves_to_knowledge(filename: str, total_results: int=100) -> dict:
    with open(filename, "r") as f:
        i: int = 0
        table_moves: list = []
        while True:
            moves_str: str = f.readline()
            if (total_results is not None and i >= total_results) or len(moves_str) == 0:
                break
            moves = moves_str.split(",")
            winner = moves.pop().replace("\n", "")
            if len(moves) > 0:
                table_moves.append((moves, winner))
            i += 1
        table = build_knowledge(table_moves)
    return table


def build_knowledge(table_moves: list) -> dict:
    table = {}
    for moves, winner in table_moves:
        path = [winner]
        for move in moves:
            path.append(move)
            key = ",".join(path)
            table[key] = table.get(key, 0) + 1
    return table


def get_sts(table: dict, player: int, moves: list):
    moves_str = ",".join(moves)
    adversarial = Player.O if player == Player.X else Player.X
    won = table.get(f"{player},{moves_str}", 0)
    lost = table.get(f"{adversarial},{moves_str}", 0)
    draw = table.get(f"{Player.U},{moves_str}", 0)
    total = won + lost + draw
    sts = {"won": 0, "lost": 0, "draw": 0}
    if total > 0:
        sts["won"] = won / total 
        sts["lost"] = lost / total
        sts["draw"] = draw / total
    return sts


def get_sts_kill_move(plays, player, move:str):
    total = 0
    lost = 0
    won = 0
    for w, base_moves in plays.items():
        for base_move in base_moves:
            if move in base_move and w == str(player):
                won += 1
            elif move in base_move and w != str(player):
                lost += 1

    sts = {"won": 0, "lost": 0}
    total = won + lost
    if total > 0:
        sts["won"] = won / total 
        sts["lost"] = lost / total
    
    return sts


def next_moves(moves_made: list, steps: int = 1) -> list:
    moves_pool = permutations(product([0,1,2], repeat=2), 1)
    moves_pool_set = set([f"{move[0]}{move[1]}" for moves in moves_pool for move in moves])
    moves_left = moves_pool_set.difference(set(moves_made))
    moves_left = [moves_made + list(move) for move in permutations(moves_left, steps)]
    return moves_left


def play_sts(plays, move):
    moves = next_moves(move, 1)
    best_moves = []
    top_moves = []
    for posible_moves in moves:
        posible_move_str = ",".join(posible_moves)
        sts = get_sts_kill_move(plays, Player.X, posible_move_str)
        best_moves.append((sts, posible_moves))
    top_moves = heapq.nlargest(8, best_moves, key=lambda x: x[0]["won"])
    best_score = top_moves[0][0]["won"]
    choice_list: list = []
    for move in top_moves:
        if move[0]["won"] == best_score:
            choice_list.append(move)
    ramdom_choice = choice(choice_list)
    return ramdom_choice


def best_play_game(table: dict, plays, stop_move: int = 1, moves: list = [], print_board_on_terminal: bool = True) -> list:
    while True:
        active_player = Player.X if len(moves) % 2 == 0 else Player.O
        best_play = play_sts(plays, moves)
        moves = best_play[1]
        if active_player == Player.O:
            if "11" not in moves and choice([0, 1]) == 1:
                moves.pop()
                moves.append("11")
        
        if best_play[0]["won"] == 0:
            moves.pop()
            moves = only_block_move(table, moves)
        
        board = moves_to_board(moves)
        winner = board.winner()
        if print_board_on_terminal is True:
            board.print()
        if len(moves) == stop_move or winner != Player.U:
            moves.append(str(winner))
            return moves


def only_block_move(table: dict, moves: list):
    board = Board()
    block_move = moves[:]
    for player_turn, position in enumerate(moves):
        i = int(position[0])
        j = int(position[1])
        active_player = Player.X if player_turn % 2 == 0 else Player.O
        board.add_move(i, j, active_player)
    winner_player, vector, name_position = board.check_board(status="to_win")

    if name_position == "":
        active_player = Player.X if len(moves) % 2 == 0 else Player.O
        next_moves_list = next_moves(moves, steps=1)
        posible_moves = []
        last_move = set([])
        for next_move in next_moves_list:
            sts = get_sts(table, active_player, next_move)
            if next_move[len(moves)] not in last_move:
                posible_moves.append((sts, next_move[len(moves)]))
                last_move.add(next_move[len(moves)])
        
        posible_moves = heapq.nlargest(2, posible_moves, key=lambda x: x[0]["won"])
        posible_moves = choice(posible_moves)
        block_move.append(posible_moves[1])
    else:
        i = 0
        while i <= 2:
            if vector[i] == Player.U:
                if name_position.startswith("r"):
                    index =  f"{name_position[1]}{i}"
                elif name_position.startswith("c"):
                    index = f"{i}{name_position[1]}"
                elif name_position == "d0":
                    index = f"{i}{i}"
                else:
                    index = f"{i}{2-i}"

                block_move.append(index)
                break
            i += 1
    return block_move


def rotate(moves: list) -> int:
    c = ["00", "02", "22", "20"]
    m = ["01", "12", "21", "10"]

    if moves[0] != "11":
        rotation_move = moves[0]
    else:
        rotation_move = moves[1]
    
    try:
        index = c.index(rotation_move)
    except ValueError:
        index = m.index(rotation_move)

    rotation: int = 4 - index
    return rotation


def rotate_moves(rotation: int, moves: list) -> list:
    if rotation == 4:
        return moves
    
    c = ["00", "02", "22", "20"]
    m = ["01", "12", "21", "10"]
    rotated_moves: list = []
    
    for move in moves:
        if move != "11":
            try:
                index = c.index(move)
                v = c
            except ValueError:
                index = m.index(move)
                v = m
            rotation_index = (rotation + index) % 4
            rotated_moves.append(v[rotation_index])
        else:
            rotated_moves.append("11")
    return rotated_moves

# base_reflections
def reflection(moves: list) -> dict:
    reflected_moves = {}

    reflection_h = ["00", "20", "01", "21", "02", "22"]
    reflection_v = ["00", "02", "10", "12", "20", "22"]
    reflection_d = ["00", "22", "10", "21", "01", "12"]
    reflection_d2 = ["02", "20", "01", "10", "12", "21"]

    for i, reflection in enumerate([reflection_h, reflection_v, reflection_d, reflection_d2]):
        i = str(i)
        reflected_moves[i] = []
        for move in moves:
            if move in reflection:
                index = reflection.index(move)
                if index % 2 == 1:
                    move_tmp = reflection[index - 1]
                else:
                    move_tmp = reflection[index + 1]
                reflected_moves[i].append(move_tmp)
            else:
                reflected_moves[i].append(move)

    return reflected_moves


# convert a move into its minimun reflected value
def reflection_moves(moves: list) -> str:
    reflected_moves = all_reflections(moves)
    min_reflected_moves = min(reflected_moves)
    return min_reflected_moves
    

# reflections base and its reflections
def all_reflections(moves: list) -> set:
    reflected_moves: set = set([])
    base_reflected_moves: list = list(reflection(moves).values())
    while len(base_reflected_moves) > 0:
        base_reflection_move = base_reflected_moves.pop()
        base_reflection_move_str = ",".join(base_reflection_move)
        if base_reflection_move_str not in reflected_moves:
            reflected_moves.add(base_reflection_move_str)
            for derived_reflections in list(reflection(base_reflection_move).values()):
                base_reflected_moves.insert(0, derived_reflections)

    return reflected_moves


def reverse_order(moves: list) -> set:
    reversed_order: set = set([])
    for move in moves:
        move_list = move.split(",")
        move_list.reverse()
        reversed_order.add(",".join(move_list))
    return reversed_order


def read_results_killer_moves(filename: str, use_reflections: bool = True) -> dict:
    plays: dict = defaultdict(set)
    with open(filename, "r") as f:
        while True:
            moves_str = f.readline()
            if len(moves_str) > 0:
                winner_moves = moves_str.split(",")
                winner_player = winner_moves[-1].replace("\n", "")
                if use_reflections is True:
                    all_reflex = all_reflections(winner_moves[:-1])
                    all_reflex_reverse = reverse_order(all_reflex)
                    all_reflexed_moves = all_reflex.union(all_reflex_reverse)
                else:
                    all_reflexed_moves = set([",".join(winner_moves[:-1])])
                plays[winner_player] = plays[winner_player].union(all_reflexed_moves)
            else:
                break
    return plays


def save_best_plays(table: dict, plays: dict, filename: str, number_games: int = 100):
    games = set([])
    for _ in range(number_games):
        game = best_play_game(table, plays, stop_move=9, print_board_on_terminal=False)
        game_str = ",".join(game)
        if game_str not in games:
            games.add(game_str)
    
    print(len(games))
    with open(filename, "w") as f:
        for game_str in games:
            f.write(game_str)
            f.write("\n")


def get_base_plays(filename: str):
    moves_set = set([])
    checked_reflection = set([])
    with open(filename, "r") as f:
        while True:
            moves_str = f.readline()
            moves = moves_str.split(",")
            if moves_str == "":
                break 
            elif moves[-1].replace("\n", "") == str(Player.U):
                reflected_zero_move = all_reflections(moves[:3])
                if ",".join(moves[:3]) not in checked_reflection:
                    checked_reflection = checked_reflection.union(reflected_zero_move)
                    moves_set.add(",".join(moves[:3]))
    
    grid_lines = []
    for moves_str in moves_set:
        reflected_moves_str = reflection_moves(moves_str.split(","))
        board = moves_to_board(reflected_moves_str.split(","))
        grid_lines.append(get_grid_lines(board))

    grid_lines_chunks = []
    chunk_size = 4
    for i in range(0, len(grid_lines), chunk_size):
        grid_lines_chunks.append(grid_lines[i:i + chunk_size])

    spc = " " * 10
    for grid_lines_chunk in grid_lines_chunks:
        for i in range(len(grid_lines[0])):
            print(f"{grid_lines_chunk[0][i]}{spc}{grid_lines_chunk[1][i]}{spc}{grid_lines_chunk[2][i]}{spc}{grid_lines_chunk[3][i]}")


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    parser = argparse.ArgumentParser(
        prog="tictactoe",
        description="program to find the tictactoe don't lose pattern."
    )

    parser.add_argument("--generate-all-moves", required=False, action="store_true")
    parser.add_argument("--game-patterns", required=False, action="store_true")
    parser.add_argument("--run-best-plays", required=False, type=int)
    args = parser.parse_args()

    game_patterns_filename = config["filenames"]["game_patterns"]
    positions_filename = config["filenames"]["all_positions_play"]
    positions_sts_filename = config["filenames"]["all_positions_sts"]
    positions_no_duplicated_filename = config["filenames"]["positions_no_duplicated"]
    positions_no_symmetries_filename = config["filenames"]["positions_no_symmetries"]
    positions_win_or_lose_filename = config["filenames"]["positions_win_or_lose"]

    if args.generate_all_moves:
        clean_play_positions(filename=positions_filename)
        print("playing all games...")
        play()
        remove_duplicated_moves(input_filename=positions_filename, 
                               output_filename=positions_no_duplicated_filename)
        print("removing duplicated games...")
        remove_symmetric_moves(input_filename=positions_no_duplicated_filename, 
                               output_filename=positions_no_symmetries_filename)
        print("removing symmetric games...")
        table = moves_to_knowledge(positions_no_symmetries_filename, total_results=None)
        print("saving last moves...")
        save_win_or_lose_moves(table, 
                               input_filename=positions_no_symmetries_filename, 
                               output_filename=positions_win_or_lose_filename)
    elif args.run_best_plays:
        table = moves_to_knowledge(positions_no_symmetries_filename, total_results=None)
        plays = read_results_killer_moves(positions_win_or_lose_filename, use_reflections=True)
        save_best_plays(table, plays, game_patterns_filename, number_games=args.run_best_plays)
    elif args.game_patterns:
        get_base_plays(game_patterns_filename)
