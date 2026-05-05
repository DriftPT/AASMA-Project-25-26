import random
from typing import Tuple, List


Position = Tuple[int, int]


def manhattan_distance(a: Position, b: Position) -> int:
    """
    Manhattan distance for grid worlds.

    Example:
    (0, 0) to (2, 3) = 5
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_towards(current: Position, target: Position) -> Position:
    """
    Returns a position one step closer to the target.
    """
    x, y = current
    tx, ty = target

    possible_moves: List[Position] = []

    if tx > x:
        possible_moves.append((x + 1, y))
    elif tx < x:
        possible_moves.append((x - 1, y))

    if ty > y:
        possible_moves.append((x, y + 1))
    elif ty < y:
        possible_moves.append((x, y - 1))

    if possible_moves:
        return random.choice(possible_moves)

    return current


def move_away_from(current: Position, danger: Position) -> Position:
    """
    Returns a position one step away from a dangerous position.
    """
    x, y = current
    dx, dy = danger

    possible_moves: List[Position] = []

    if dx >= x:
        possible_moves.append((x - 1, y))
    else:
        possible_moves.append((x + 1, y))

    if dy >= y:
        possible_moves.append((x, y - 1))
    else:
        possible_moves.append((x, y + 1))

    return random.choice(possible_moves)