from pathlib import Path

from src.model import ZombieSurvivalModel
from src.utils import manhattan_distance
from config.config import (
    GRID_HEIGHT,
    GRID_WIDTH,
    MAX_STEPS,
    NUM_OBSTACLES,
    NUM_ZOMBIES,
)


SEED = 404
STEPS_TO_DEBUG = 200
OUTPUT_FILE = Path(f"debug_seed_{SEED}.out")


model = ZombieSurvivalModel(
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    num_zombies=NUM_ZOMBIES,
    num_obstacles=NUM_OBSTACLES,
    max_steps=MAX_STEPS,
    team_mode="baseline",
    seed=SEED,
)


def write_step(file, model, step_num):
    file.write(f"\n{'=' * 60}\n")
    file.write(f"STEP {step_num}\n")
    file.write(f"{'=' * 60}\n")

    survivors = model.get_alive_survivors()
    zombies = model.get_alive_zombies()

    file.write("\n--- SURVIVORS ---\n")

    for s in survivors:
        teammates = [t for t in survivors if t != s]

        dist_to_teammates = {
            type(t).__name__: manhattan_distance(s.pos, t.pos)
            for t in teammates
        }

        nearest_z = model.nearest_zombie(s.pos)
        dist_to_zombie = (
            manhattan_distance(s.pos, nearest_z.pos)
            if nearest_z
            else "N/A"
        )

        dist_to_safe = manhattan_distance(s.pos, model.safe_zone_pos)

        action = s.last_action.value if s.last_action else "none"

        file.write(
            f"  {type(s).__name__:<15} "
            f"pos={str(s.pos):<10} "
            f"hp={s.health} | "
            f"action={action:<8} | "
            f"dist_safe={dist_to_safe:<4} | "
            f"dist_zombie={dist_to_zombie:<4} | "
            f"dist_teammates={dist_to_teammates}\n"
        )

    file.write("\n--- ZOMBIES ---\n")

    for z in zombies:
        nearest_s = (
            min(
                survivors,
                key=lambda s: manhattan_distance(z.pos, s.pos),
            )
            if survivors
            else None
        )

        dist_to_nearest = (
            manhattan_distance(z.pos, nearest_s.pos)
            if nearest_s
            else "N/A"
        )

        target_name = type(nearest_s).__name__ if nearest_s else "N/A"

        file.write(
            f"  Zombie pos={str(z.pos):<10} "
            f"hp={z.health} | "
            f"targeting={target_name:<15} "
            f"dist={dist_to_nearest}\n"
        )

    file.write(f"\n  Cooperation events: {model.cooperation_events}\n")
    file.write(f"  Finished: {model.finished} | Success: {model.is_successful()}\n")


with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    file.write("Zombie Survival Debug Output\n")
    file.write(f"Seed: {SEED}\n")
    file.write(f"Team mode: baseline\n")
    file.write(f"Grid: {GRID_WIDTH}x{GRID_HEIGHT}\n")
    file.write(f"Zombies: {NUM_ZOMBIES}\n")
    file.write(f"Obstacles: {NUM_OBSTACLES}\n")
    file.write(f"Max steps: {MAX_STEPS}\n")
    file.write(f"Debug steps: {STEPS_TO_DEBUG}\n")

    write_step(file, model, 0)

    for i in range(1, STEPS_TO_DEBUG + 1):
        model.step()
        write_step(file, model, i)

        if model.finished:
            file.write(f"\n>>> SIMULATION ENDED at step {i}\n")
            file.write(f">>> Success: {model.is_successful()}\n")
            break

print(f"Debug output written to: {OUTPUT_FILE.resolve()}")