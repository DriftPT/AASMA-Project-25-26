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
        if self.pos == self.model.safe_zone_pos:
            self.reached_safe_zone = True


class ScoutAgent(SurvivorAgent):
    """
    Scout:
    - focuses on moving safely toward the safe zone;
    - avoids zombies when they are too close;
    - does not actively fight unless necessary.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Scout", symbol="C")

    def step(self):
        if not self.alive or self.model.finished:
            return

        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is not None:
            distance = manhattan_distance(self.pos, nearest_zombie.pos)

            if distance <= 1:
                new_pos = move_away_from(self.pos, nearest_zombie.pos)
                self.move_to(new_pos)
                self.check_safe_zone()
                return

        new_pos = move_towards(self.pos, self.model.safe_zone_pos)
        self.move_to(new_pos)
        self.check_safe_zone()


class DefenderAgent(SurvivorAgent):
    """
    Defender:
    - attacks nearby zombies;
    - moves toward zombies that are close;
    - otherwise stays near teammates.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Defender", symbol="D")

    def step(self):
        if not self.alive or self.model.finished:
            return

        attacked = self.attack_nearby_zombie()

        if attacked:
            return

        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is not None:
            distance = manhattan_distance(self.pos, nearest_zombie.pos)

            if distance <= 4:
                new_pos = move_towards(self.pos, nearest_zombie.pos)
                self.move_to(new_pos)
                self.check_safe_zone()
                return

        nearest_ally = self.model.nearest_survivor(self.pos, exclude=self)

        if nearest_ally is not None:
            new_pos = move_towards(self.pos, nearest_ally.pos)
            self.move_to(new_pos)
            self.check_safe_zone()
            return

        self.last_action = Action.WAIT


class SupportAgent(SurvivorAgent):
    """
    Support:
    - heals injured teammates;
    - moves toward injured teammates;
    - otherwise stays close to the group.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Support", symbol="U")

    def step(self):
        if not self.alive or self.model.finished:
            return

        healed = self.heal_nearby_survivor()

        if healed:
            return

        injured_survivors = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self and survivor.health < survivor.max_health
        ]

        if injured_survivors:
            target = min(
                injured_survivors,
                key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
            )

            new_pos = move_towards(self.pos, target.pos)
            self.move_to(new_pos)
            self.check_safe_zone()
            return

        nearest_ally = self.model.nearest_survivor(self.pos, exclude=self)

        if nearest_ally is not None:
            new_pos = move_towards(self.pos, nearest_ally.pos)
            self.move_to(new_pos)
            self.check_safe_zone()
            return

        self.last_action = Action.WAIT


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
        if not self.alive or self.model.finished:
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
        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is not None:
            distance = manhattan_distance(self.pos, nearest_zombie.pos)

            if distance <= 1:
                new_pos = move_away_from(self.pos, nearest_zombie.pos)
                self.move_to(new_pos)
                return

        new_pos = move_towards(self.pos, self.model.safe_zone_pos)
        self.move_to(new_pos)

    def defender_behaviour(self):
        attacked = self.attack_nearby_zombie()

        if attacked:
            return

        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is not None:
            new_pos = move_towards(self.pos, nearest_zombie.pos)
            self.move_to(new_pos)
            return

        self.last_action = Action.WAIT

    def support_behaviour(self):
        healed = self.heal_nearby_survivor()

        if healed:
            return

        injured_survivors = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self and survivor.health < survivor.max_health
        ]

        if injured_survivors:
            target = min(
                injured_survivors,
                key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
            )

            new_pos = move_towards(self.pos, target.pos)
            self.move_to(new_pos)
            return

        nearest_ally = self.model.nearest_survivor(self.pos, exclude=self)

        if nearest_ally is not None:
            new_pos = move_towards(self.pos, nearest_ally.pos)
            self.move_to(new_pos)
            return

        self.last_action = Action.WAIT