from src.model import ZombieSurvivalModel
from src.agents.rl_policy import RLPolicy

from config.config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    NUM_ZOMBIES,
    NUM_OBSTACLES,
    MAX_STEPS,
    RANDOM_SEED,
)

from src.metrics import collect_episode_result, summarize_results


# ==============================
# RL CONFIGURATION
# ==============================

ALGORITHM = "sarsa" #ou q_learning
TRAIN_EPISODES = 1500
TEST_EPISODES = 500
TRAIN_EPSILON = 1
EPSILON_DECAY = 0.995
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
    policy = RLPolicy(
        actions=ADAPTIVE_ACTIONS,
        algorithm=ALGORITHM,
        alpha=0.2,
        gamma=0.9,
        epsilon=TRAIN_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        training=True,
    )

    training_modes = ["adaptive_replaces_scout","adaptive_replaces_defender","adaptive_replaces_support","adaptive_adaptive_adaptive"]

    for episode in range(train_episodes):
        team_mode = training_modes[episode % len(training_modes)]

        run_episode(
            team_mode=team_mode,
            seed=RANDOM_SEED + TEST_EPISODES + episode,
            adaptive_policy=policy,
        )

        policy.decay_epsilon()
    return policy

def run_experiment(team_mode: str, episodes: int = TEST_EPISODES, adaptive_policy=None,):
    results = []

    for i in range(episodes):
        result = run_episode(
            team_mode=team_mode,
            seed=RANDOM_SEED + i,
            adaptive_policy=adaptive_policy,
        )

        results.append(result)

    return summarize_results(results)


def run_all_experiments():
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