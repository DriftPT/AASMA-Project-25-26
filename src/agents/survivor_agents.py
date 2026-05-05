from enum import Enum
from typing import Optional, Dict

from mesa import Agent

from src.utils.utils import manhattan_distance, move_towards, move_away_from


class Action(Enum):
    MOVE = "move"
    ATTACK = "attack"
    HEAL = "heal"
    WAIT = "wait"


class SurvivorAgent(Agent):
    """
    Base class for all survivor agents.
    """

    def __init__(self, model, role_name: str, symbol: str):
        super().__init__(model)

        self.role_name = role_name
        self.symbol = symbol

        self.health = self.model.initial_health
        self.max_health = self.model.initial_health
        self.alive = True

        self.last_action: Optional[Action] = Action.WAIT
        self.reached_safe_zone = False

    def step(self):
        raise NotImplementedError

    def take_damage(self, amount: int):
        self.health -= amount

        if self.health <= 0:
            self.alive = False

            if self.pos is not None:
                self.model.grid.remove_agent(self)

            self.remove()

    def move_to(self, new_pos):
        if self.model.can_move_to(new_pos):
            self.model.grid.move_agent(self, new_pos)
            self.last_action = Action.MOVE
        else:
            self.last_action = Action.WAIT

    def attack_nearby_zombie(self) -> bool:
        for zombie in self.model.get_alive_zombies():
            if manhattan_distance(self.pos, zombie.pos) == 1:
                zombie.health -= 1
                self.last_action = Action.ATTACK

                if zombie.health <= 0:
                    zombie.alive = False

                    if zombie.pos is not None:
                        self.model.grid.remove_agent(zombie)

                    zombie.remove()

                return True

        return False

    def heal_nearby_survivor(self) -> bool:
        for survivor in self.model.get_alive_survivors():
            if survivor == self:
                continue

            if survivor.health < survivor.max_health:
                if manhattan_distance(self.pos, survivor.pos) <= 1:
                    survivor.health += 1
                    self.model.cooperation_events += 1
                    self.last_action = Action.HEAL
                    return True

        return False

    def check_safe_zone(self):
        if self.pos in self.model.safe_zone_positions:
            self.reached_safe_zone = True
    
    def stay_if_reached_safe_zone(self) -> bool:
        """
        If the survivor has reached the safe zone, it stays there.
        This prevents agents from leaving the safe zone while waiting
        for the rest of the team.
        """
        if self.pos in self.model.safe_zone_positions:
            self.reached_safe_zone = True
            self.last_action = Action.WAIT
            return True

        return False

    # ======================================================
    # Shared team behaviour helpers
    # ======================================================

    def get_team_goal(self):
        """
        Returns the current team goal.

        The Scout can create/update this shared intention.
        If no Scout has acted yet, the default goal is still the safe zone.
        """
        return getattr(self.model, "team_goal_pos", self.model.safe_zone_pos)

    def call_team_to_safe_zone(self):
        """
        Simple implicit communication mechanism.

        When the Scout decides that the best plan is to reach the safe zone,
        it stores that goal in the model. Other agents can read it and follow.
        """
        self.model.team_goal_pos = self.model.safe_zone_pos
        self.model.team_leader_pos = self.pos

    def nearby_zombie(self, max_distance: int):
        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is None:
            return None

        if manhattan_distance(self.pos, nearest_zombie.pos) <= max_distance:
            return nearest_zombie

        return None

    def teammate_threatened_by_zombie(self, max_distance: int = 1):
        """
        Returns a zombie that is threatening any alive teammate.
        Used mainly by the Defender.
        """
        threatened_zombies = []

        for zombie in self.model.get_alive_zombies():
            for survivor in self.model.get_alive_survivors():
                if survivor == self:
                    continue

                if manhattan_distance(zombie.pos, survivor.pos) <= max_distance:
                    threatened_zombies.append(zombie)
                    break

        if not threatened_zombies:
            return None

        return min(
            threatened_zombies,
            key=lambda zombie: manhattan_distance(self.pos, zombie.pos)
        )

    def move_safely_towards(self, target_pos):
        """
        Moves one cell toward a target, but avoids moves that place the agent
        next to a zombie when possible.
        """
        x, y = self.pos
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

        valid_moves = [pos for pos in candidates if self.model.can_move_to(pos)]

        if not valid_moves:
            self.last_action = Action.WAIT
            return

        def zombie_risk(pos):
            zombies = self.model.get_alive_zombies()
            if not zombies:
                return 0
            return min(manhattan_distance(pos, zombie.pos) for zombie in zombies)

        current_distance = manhattan_distance(self.pos, target_pos)

        progress_moves = [
            pos for pos in valid_moves
            if manhattan_distance(pos, target_pos) < current_distance
        ]

        if progress_moves:
            valid_moves = progress_moves

        # Prefer progress toward the target and avoid zombie-adjacent cells.
        best_distance = min(manhattan_distance(pos, target_pos) for pos in valid_moves)
        best_moves = [
            pos for pos in valid_moves
            if manhattan_distance(pos, target_pos) == best_distance
        ]

        safe_moves = [pos for pos in best_moves if zombie_risk(pos) > 1]

        if safe_moves:
            best_moves = safe_moves

        new_pos = self.model.random.choice(best_moves)
        self.move_to(new_pos)

    def follow_team_goal(self):
        self.move_safely_towards(self.get_team_goal())
        self.check_safe_zone()


class ScoutAgent(SurvivorAgent):
    """
    Scout:
    - leads the team toward the safe zone;
    - calls teammates to follow the safe-zone objective;
    - avoids zombies when they are too close;
    - only stops progressing when survival requires it.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Scout", symbol="C")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        # The Scout acts as a leader: it announces the shared objective.
        self.call_team_to_safe_zone()

        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is not None:
            distance = manhattan_distance(self.pos, nearest_zombie.pos)

            # If a zombie is adjacent, escaping has priority over moving forward.
            if distance <= 1:
                new_pos = move_away_from(self.pos, nearest_zombie.pos)

                if self.model.can_move_to(new_pos):
                    self.move_to(new_pos)
                else:
                    # If escaping is blocked, attack only as a last resort.
                    attacked = self.attack_nearby_zombie()
                    if not attacked:
                        self.last_action = Action.WAIT

                self.check_safe_zone()
                return

        self.follow_team_goal()


class DefenderAgent(SurvivorAgent):
    """
    Defender:
    - protects the team from immediate threats;
    - attacks adjacent zombies;
    - intercepts zombies that are threatening teammates;
    - otherwise follows the Scout/team goal toward the safe zone.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Defender", symbol="D")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        # Direct danger: fight because there is no safe alternative.
        attacked = self.attack_nearby_zombie()
        if attacked:
            self.check_safe_zone()
            return

        # Protect teammates only when a zombie is an immediate threat.
        threatening_zombie = self.teammate_threatened_by_zombie(max_distance=1)
        if threatening_zombie is not None:
            new_pos = move_towards(self.pos, threatening_zombie.pos)
            self.move_to(new_pos)
            self.check_safe_zone()
            return

        # Otherwise, the defender should not chase zombies forever.
        self.follow_team_goal()


class SupportAgent(SurvivorAgent):
    """
    Support:
    - heals nearby injured teammates;
    - helps injured teammates only when they are close enough;
    - otherwise follows the Scout/team goal toward the safe zone.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Support", symbol="S")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        healed = self.heal_nearby_survivor()
        if healed:
            self.check_safe_zone()
            return

        injured_survivors = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self and survivor.health < survivor.max_health
        ]

        # Do not abandon the safe-zone objective for a far-away injured agent.
        # Support only diverts if the injured teammate is close enough to help.
        close_injured_survivors = [
            survivor for survivor in injured_survivors
            if manhattan_distance(self.pos, survivor.pos) <= 3
        ]

        if close_injured_survivors:
            target = min(
                close_injured_survivors,
                key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
            )

            self.move_safely_towards(target.pos)
            self.check_safe_zone()
            return

        self.follow_team_goal()


class AdaptiveAgent(SurvivorAgent):
    """
    Adaptive ad hoc agent.

    Current version:
    - observation-based;
    - rule-based role inference;
    - adapts by choosing the role that appears to be missing.

    Later, this can be replaced or extended with Reinforcement Learning.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Adaptive", symbol="A")

        self.teammate_scores: Dict[int, Dict[str, int]] = {}

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.observe_teammates()

        team_roles = self.classify_team()
        missing_role = self.choose_missing_role(team_roles)

        if missing_role == "defender":
            self.defender_behaviour()
        elif missing_role == "support":
            self.support_behaviour()
        else:
            self.scout_behaviour()

        self.check_safe_zone()

    def observe_teammates(self):
        """
        Observes teammates and updates role scores.

        The adaptive agent does not know teammate roles directly.
        It estimates them based on visible behaviour.
        """
        for teammate in self.model.get_alive_survivors():
            if teammate == self:
                continue

            if teammate.unique_id not in self.teammate_scores:
                self.teammate_scores[teammate.unique_id] = {
                    "scout": 0,
                    "defender": 0,
                    "support": 0,
                }

            scores = self.teammate_scores[teammate.unique_id]

            if teammate.last_action == Action.ATTACK:
                scores["defender"] += 2

            if teammate.last_action == Action.HEAL:
                scores["support"] += 2

            distance_to_safe_zone = manhattan_distance(
                teammate.pos,
                self.model.safe_zone_pos
            )

            if distance_to_safe_zone <= 4:
                scores["scout"] += 1

            nearest_zombie = self.model.nearest_zombie(teammate.pos)

            if nearest_zombie is not None:
                distance_to_zombie = manhattan_distance(
                    teammate.pos,
                    nearest_zombie.pos
                )

                if teammate.last_action == Action.MOVE and distance_to_zombie > 2:
                    scores["scout"] += 1

    def classify_team(self):
        roles = {
            "scout": 0,
            "defender": 0,
            "support": 0,
        }

        for _, scores in self.teammate_scores.items():
            predicted_role = max(scores, key=scores.get)
            roles[predicted_role] += 1

        return roles

    def choose_missing_role(self, roles):
        return min(roles, key=roles.get)

    def scout_behaviour(self):
        # If the adaptive agent is acting as the Scout, it also calls the team.
        self.call_team_to_safe_zone()

        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is not None:
            distance = manhattan_distance(self.pos, nearest_zombie.pos)

            if distance <= 1:
                new_pos = move_away_from(self.pos, nearest_zombie.pos)

                if self.model.can_move_to(new_pos):
                    self.move_to(new_pos)
                else:
                    attacked = self.attack_nearby_zombie()
                    if not attacked:
                        self.last_action = Action.WAIT
                return

        self.follow_team_goal()

    def defender_behaviour(self):
        attacked = self.attack_nearby_zombie()

        if attacked:
            return

        threatening_zombie = self.teammate_threatened_by_zombie(max_distance=1)
        if threatening_zombie is not None:
            new_pos = move_towards(self.pos, threatening_zombie.pos)
            self.move_to(new_pos)
            return

        self.follow_team_goal()

    def support_behaviour(self):
        healed = self.heal_nearby_survivor()

        if healed:
            return

        injured_survivors = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self and survivor.health < survivor.max_health
        ]

        close_injured_survivors = [
            survivor for survivor in injured_survivors
            if manhattan_distance(self.pos, survivor.pos) <= 3
        ]

        if close_injured_survivors:
            target = min(
                close_injured_survivors,
                key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
            )

            self.move_safely_towards(target.pos)
            return

        self.follow_team_goal()