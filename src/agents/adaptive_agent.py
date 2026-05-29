from src.agents.survivor_agents import (
    SurvivorAgent,
    ScoutAgent,
    DefenderAgent,
    SupportAgent,
)
from src.agents.rl_policy import RLPolicy
from src.utils import manhattan_distance


class AdaptiveAgent(SurvivorAgent):

    ACTIONS = ("scout", "defender", "support")

    def __init__(self, model):
        super().__init__(model, role_name="Adaptive", symbol="A")

        self.current_role = "Adaptive"
        self.next_action = None  # New property to support the on-policy flow (SARSA)

        if getattr(model, "adaptive_policy", None) is not None:
            self.policy = model.adaptive_policy
        else:
            self.policy = RLPolicy(
                actions=self.ACTIONS,
                algorithm="q_learning",  # Default
                epsilon=0.0,
                training=False,
            )

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        state_before = self.get_state()

        # If using SARSA and the action was already chosen at the end of the previous step, reuse it
        if self.next_action is not None:
            action = self.next_action
            self.next_action = None 
        else:
            action = self.policy.choose_action(
                state=state_before,
                rng=self.model.random,
            )

        old_health = self.health
        old_distance_to_safe = self.distance_to_nearest_safe_zone()
        old_attack_events = self.model.attack_events
        old_heal_events = self.model.heal_events
        old_scan_events = self.model.scan_events
        old_safe_discovered = self.model.safe_zone_discovered
        old_teammate_count = len([s for s in self.model.get_alive_survivors() if s is not self])

        self.perform_learned_action(action)

        self.check_safe_zone()

        reward = self.compute_reward(
            old_health=old_health,
            old_distance_to_safe=old_distance_to_safe,
            old_attack_events=old_attack_events,
            old_heal_events=old_heal_events,
            old_scan_events=old_scan_events,
            old_safe_discovered=old_safe_discovered,
            old_teammate_count=old_teammate_count,
        )

        state_after = self.get_state()
        # Choose the next action A_t+1 (required to compute the SARSA update now)
        chosen_next_action = self.policy.choose_action(state=state_after, rng=self.model.random)

        self.policy.update(
            state=state_before,
            action=action,
            reward=reward,
            next_state=state_after,
            next_action=chosen_next_action,
        )
        self.next_action = chosen_next_action

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
            self.nearby_zombie_count_bucket(),
            int(self.model.safe_zone_discovered),
            int(self._injured_teammate_in_heal_range()),
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
        """How many zombies are within vision range."""
        count = sum(
            1 for z in self.model.get_alive_zombies()
            if manhattan_distance(self.pos, z.pos) <= self.model.vision_range
        )
        if count == 0:
            return "none"
        if count <= 2:
            return "few"
        return "many"

    def _injured_teammate_in_heal_range(self) -> bool:
        """True if there is an injured teammate close enough to heal this step."""
        injured = self.closest_injured_teammate()
        if injured is None:
            return False
        return manhattan_distance(self.pos, injured.pos) <= self.model.heal_range

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
        old_attack_events,
        old_heal_events,
        old_scan_events,
        old_safe_discovered,
        old_teammate_count,
    ):
        """
        Reward used by Q-learning.

        Positive reward:
        - getting closer to the safe zone;
        - discovering the safe zone;
        - attacking zombies, healing teammates, scanning for the safe zone;
        - reaching the safe zone;
        - protecting teammates from nearby zombies;
        - team success.

        Negative reward:
        - wasting time;
        - moving away from the safe zone;
        - losing health;
        - dying;
        - teammate death.

        Note: scan only rewards while the safe zone is still undiscovered.
        """

        reward = -0.1

        new_distance_to_safe = self.distance_to_nearest_safe_zone()

        if new_distance_to_safe < old_distance_to_safe:
            reward += 1.0
        elif new_distance_to_safe > old_distance_to_safe:
            reward -= 0.8

        if not old_safe_discovered and self.model.safe_zone_discovered:
            reward += 3.0

        if self.model.attack_events > old_attack_events:
            reward += 4.0

        if self.model.heal_events > old_heal_events:
            reward += 5.0

        # scan only rewarded while safe zone still unknown
        if self.model.scan_events > old_scan_events and not old_safe_discovered:
            reward += 1.5

        if self.health < old_health:
            reward -= 2.0

        if self.pos in self.model.safe_zone_positions:
            reward += 5.0

        # reward for keeping zombies away from teammates
        alive_zombies = self.model.get_alive_zombies()
        alive_teammates = [s for s in self.model.get_alive_survivors() if s is not self]
        if alive_zombies and alive_teammates:
            min_threat = min(
                manhattan_distance(s.pos, z.pos)
                for s in alive_teammates
                for z in alive_zombies
            )
            if min_threat > self.model.attack_range:
                reward += 0.5

        new_teammate_count = len([s for s in self.model.get_alive_survivors() if s is not self])
        if new_teammate_count < old_teammate_count:
            reward -= 12.0

        if self.model.is_successful():
            reward += 20.0

        if not self.alive:
            reward -= 20.0

        return reward