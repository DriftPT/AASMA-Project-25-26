from src.model import ZombieSurvivalModel

from config.config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    NUM_ZOMBIES,
    NUM_OBSTACLES,
    MAX_STEPS,
    EPISODES,
    RANDOM_SEED,
)

from src.metrics import collect_episode_result, summarize_results


TEAM_MODES = {
    "Baseline: Scout + Defender + Support": "baseline",
    "Adaptive replaces Scout": "adaptive_replaces_scout",
    "Adaptive replaces Defender": "adaptive_replaces_defender",
    "Adaptive replaces Support": "adaptive_replaces_support",
}


def run_episode(team_mode: str, seed: int):
    model = ZombieSurvivalModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode=team_mode,
        seed=seed,
    )

    while not model.finished:
        model.step()

    return collect_episode_result(model)


def run_experiment(team_mode: str, episodes: int = EPISODES):
    results = []

    for i in range(episodes):
        result = run_episode(
            team_mode=team_mode,
            seed=RANDOM_SEED + i,
        )

        results.append(result)

    return summarize_results(results)


def run_all_experiments():
    all_results = {}

    for experiment_name, team_mode in TEAM_MODES.items():
        all_results[experiment_name] = run_experiment(team_mode)

    return all_results