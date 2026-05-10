import random
from typing import Iterable, List, Optional, Tuple, TypeVar


Position = Tuple[int, int]
T = TypeVar("T")


def manhattan_distance(a: Position, b: Position) -> int:
    """Manhattan distance for grid worlds."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def grid_neighbours(pos: Position) -> List[Position]:
    """Returns the four orthogonal neighbours of a grid position."""
    x, y = pos
    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]


def closest_position(origin: Position, positions: Iterable[Position]) -> Optional[Position]:
    """Returns the closest position to origin, or None if empty."""
    positions = list(positions)
    if not positions:
        return None

    return min(positions, key=lambda pos: manhattan_distance(origin, pos))


def closest_agent(origin: Position, agents: Iterable[T]) -> Optional[T]:
    """Returns the agent closest to origin, assuming each agent has .pos."""
    agents = list(agents)
    if not agents:
        return None

    return min(agents, key=lambda agent: manhattan_distance(origin, agent.pos))


def average_position(positions: Iterable[Position], fallback: Position) -> Position:
    """Integer average position of a group of positions."""
    positions = list(positions)
    if not positions:
        return fallback

    x = sum(pos[0] for pos in positions) // len(positions)
    y = sum(pos[1] for pos in positions) // len(positions)
    return (x, y)


def move_towards(current: Position, target: Position, rng=None) -> Position:
    """Returns a position one step closer to the target."""
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
        return (rng or random).choice(possible_moves)

    return current


def move_away_from(current: Position, danger: Position, rng=None) -> Position:
    """Returns a position one step away from a dangerous position."""
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

    return (rng or random).choice(possible_moves)
