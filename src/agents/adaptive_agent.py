from src.agents.survivor_agents import (
    SurvivorAgent,
    ScoutAgent,
    DefenderAgent,
    SupportAgent,
)
from src.agents.q_learning_policy import QLearningPolicy
from src.utils import manhattan_distance


class AdaptiveAgent(SurvivorAgent):
    """
    Adaptive ad hoc agent using Q-learning.

    The Q-learning actions are high-level role behaviours:

    - "scout": reuse ScoutAgent.step(self)
    - "defender": reuse DefenderAgent.step(self)
    - "support": reuse SupportAgent.step(self)

    So the agent does not learn primitive actions directly.
    It learns which existing role policy should be used in each state.
    """

    ACTIONS = ("scout", "defender", "support")

    def __init__(self, model):
        super().__init__(model, role_name="Adaptive", symbol="A")

        # If the model received an external policy, use it.
        # This is used during training/testing so knowledge is preserved
        # between episodes.

        self.current_role = "Adaptive"

        if getattr(model, "adaptive_policy", None) is not None:
            self.policy = model.adaptive_policy
        else:
            # Fallback for visualisation / single runs — starts with empty Q-table.
            self.policy = QLearningPolicy(
                actions=self.ACTIONS,
                epsilon=0.0,
                training=False,
            )

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        state_before = self.get_state()

        action = self.policy.choose_action(
            state=state_before,
            rng=self.model.random,
        )

        old_health = self.health
        old_distance_to_safe = self.distance_to_nearest_safe_zone()
        old_cooperation_events = self.model.cooperation_events
        old_safe_discovered = self.model.safe_zone_discovered
        old_teammate_count = len([s for s in self.model.get_alive_survivors() if s is not self])

        self.perform_learned_action(action)

        self.check_safe_zone()

        reward = self.compute_reward(
            old_health=old_health,
            old_distance_to_safe=old_distance_to_safe,
            old_cooperation_events=old_cooperation_events,
            old_safe_discovered=old_safe_discovered,
            old_teammate_count=old_teammate_count,
        )

        state_after = self.get_state()

        self.policy.update(
            state=state_before,
            action=action,
            reward=reward,
            next_state=state_after,
        )

    # ==========================================================
    # Reuse existing survivor behaviours
    # ==========================================================

    def perform_learned_action(self, action):
        """
        Reuses the step() method of the existing survivor roles.

        Important:
        ScoutAgent.step(self) does not create a Scout.
        It simply runs the Scout behaviour using the AdaptiveAgent instance.
        """
        self.current_role = action.capitalize()

        if action == "scout":
            ScoutAgent.step(self)

        elif action == "defender":
            self.explore_target = None
            DefenderAgent.step(self)

        elif action == "support":
            self.explore_target = None
            SupportAgent.step(self)

    # ==========================================================
    # State representation
    # ==========================================================

    def get_state(self):
        """
        Compact state for Q-learning.

        We do not use exact grid positions because that would create too many
        possible states. Instead, we use categories that describe the situation.
        """

        return (
            self.health_bucket(),
            self.safe_distance_bucket(),
            self.zombie_distance_bucket(),
            int(self.model.safe_zone_discovered),
            int(self.closest_injured_teammate() is not None),
            int(self.team_is_too_far(max_distance=5)),
        )

    def health_bucket(self):
        if self.health <= 1:
            return "low"

        if self.health < self.max_health:
            return "medium"

        return "high"

    def safe_distance_bucket(self):
        distance = self.distance_to_nearest_safe_zone()

        if distance <= 4:
            return "near"

        if distance <= 10:
            return "medium"

        return "far"

    def zombie_distance_bucket(self):
        zombie = self.model.nearest_zombie(self.pos)

        if zombie is None:
            return "none"

        distance = manhattan_distance(self.pos, zombie.pos)

        if distance <= 1:
            return "danger"

        if distance <= 5:
            return "near"

        return "far"

    def nearby_zombie_count_bucket(self):
        """How many zombies are within vision range"""
        count = sum(
            1 for z in self.model.get_alive_zombies()
            if manhattan_distance(self.pos, z.pos) <= self.model.vision_range
        )
        if count == 0:
            return "none"
        if count <= 2:
            return "few"
        return "many"

    def distance_to_nearest_safe_zone(self):
        """Distance to the closest safe zone cell (known or unknown)."""
        if not self.model.safe_zone_positions:
            return self.model.width + self.model.height
 
        return min(
            manhattan_distance(self.pos, safe_pos)
            for safe_pos in self.model.safe_zone_positions
        )

    # ==========================================================
    # Reward function
    # ==========================================================

    def compute_reward(
        self,
        *,
        old_health,
        old_distance_to_safe,
        old_cooperation_events,
        old_safe_discovered,
        old_teammate_count,
    ):
        """
        Reward used by Q-learning.

        Positive reward:
        - getting closer to the safe zone;
        - discovering the safe zone;
        - cooperating by attacking/healing;
        - reaching the safe zone;
        - team success.

        Negative reward:
        - wasting time;
        - moving away from the safe zone;
        - losing health;
        - dying.
        """

        reward = -0.1

        new_distance_to_safe = self.distance_to_nearest_safe_zone()

        if new_distance_to_safe < old_distance_to_safe:
            reward += 1.0
        elif new_distance_to_safe > old_distance_to_safe:
            reward -= 0.5

        if not old_safe_discovered and self.model.safe_zone_discovered:
            reward += 3.0

        if self.model.cooperation_events > old_cooperation_events:
            reward += 2.0

        if self.health < old_health:
            reward -= 2.0

        if self.pos in self.model.safe_zone_positions:
            reward += 5.0

        new_teammate_count = len([s for s in self.model.get_alive_survivors() if s is not self])
        if new_teammate_count < old_teammate_count:
            reward -= 8.0

        if self.model.is_successful():
            reward += 20.0

        if not self.alive:
            reward -= 20.0
        
        return reward