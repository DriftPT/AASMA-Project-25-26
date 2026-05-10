from src.model import ZombieSurvivalModel
from src.agents.q_learning_policy import QLearningPolicy

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


# ==============================
# Q-LEARNING CONFIGURATION
# ==============================

TRAIN_EPISODES = 300
TRAIN_EPSILON = 0.25
TEST_EPSILON = 0.0

ADAPTIVE_ACTIONS = ("scout", "defender", "support")

# ==============================
# EXPERIMENT CONFIGURATIONS
# ==============================

TEAM_MODES = {
    "Baseline: Scout + Defender + Support": "baseline",
    "Adaptive replaces Scout": "adaptive_replaces_scout",
    "Adaptive replaces Defender": "adaptive_replaces_defender",
    "Adaptive replaces Support": "adaptive_replaces_support",
    "Adaptive (All Roles)": "adaptive_adaptive_adaptive"
}

# ==============================
# EPISODE EXECUTION
# ==============================

def run_episode(team_mode: str, seed: int, adaptive_policy=None):
    model = ZombieSurvivalModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode=team_mode,
        seed=seed,
        adaptive_policy=adaptive_policy,
    )

    while not model.finished:
        model.step()

    return collect_episode_result(model)

# ==============================
# TRAINING
# ==============================

def train_adaptive_agent(train_episodes: int = TRAIN_EPISODES):
    policy = QLearningPolicy(
        actions=ADAPTIVE_ACTIONS,
        alpha=0.2,
        gamma=0.9,
        epsilon=TRAIN_EPSILON,
        training=True,
    )

    training_modes = ["adaptive_replaces_scout","adaptive_replaces_defender","adaptive_replaces_support","adaptive_adaptive_adaptive"]

    for episode in range(train_episodes):
        team_mode = training_modes[episode % len(training_modes)]

        run_episode(
            team_mode=team_mode,
            seed=RANDOM_SEED + EPISODES + episode,
            adaptive_policy=policy,
        )

    return policy

def run_experiment(team_mode: str, episodes: int = EPISODES, adaptive_policy=None,):
    results = []

    for i in range(episodes):
        result = run_episode(
            team_mode=team_mode,
            seed=RANDOM_SEED + i,
            adaptive_policy=adaptive_policy,
        )

        results.append(result)

    return summarize_results(results)


def train_adaptive_agent_for_mode(team_mode: str, train_episodes: int = TRAIN_EPISODES):
    policy = QLearningPolicy(
        actions=ADAPTIVE_ACTIONS,
        alpha=0.2,
        gamma=0.9,
        epsilon=TRAIN_EPSILON,
        training=True,
    )

    for episode in range(train_episodes):
        run_episode(
            team_mode=team_mode,
            seed=RANDOM_SEED + EPISODES + episode,
            adaptive_policy=policy,
        )

    policy.set_training(False)
    policy.set_epsilon(TEST_EPSILON)

    return policy

def run_all_experiments_v1():
    all_results = {}
    print("=" * 70)
    print("Training Adaptive Agent for the modes")
    print("=" * 70)
    print()

    for experiment_name, team_mode in TEAM_MODES.items():
        if team_mode == "baseline":
            all_results[experiment_name] = run_experiment(team_mode,adaptive_policy=None)
        else:
            print(f"Training adaptive agent for mode: {team_mode}")

            adaptive_policy = train_adaptive_agent_for_mode(team_mode)

            print(
                f"Training finished. "
                f"Learned states: {adaptive_policy.number_of_learned_states()}"
            )

            all_results[experiment_name] = run_experiment(team_mode,adaptive_policy=adaptive_policy)

        print()

    return all_results

def run_all_experiments_v2():
    all_results = {}

    print("=" * 70)
    print("TRAINING ADAPTIVE AGENT WITH Q-LEARNING")
    print("=" * 70)

    adaptive_policy = train_adaptive_agent()

    print(
        f"Training finished. "
        f"Learned states: {adaptive_policy.number_of_learned_states()}"
    )
    print()
    
    adaptive_policy.set_training(False)
    adaptive_policy.set_epsilon(TEST_EPSILON)

    for experiment_name, team_mode in TEAM_MODES.items():
        if team_mode == "baseline":
            all_results[experiment_name] = run_experiment(team_mode, adaptive_policy=None)
        else:
            all_results[experiment_name] = run_experiment(team_mode, adaptive_policy=adaptive_policy)

    return all_results