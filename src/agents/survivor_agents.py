from enum import Enum
from typing import Optional

from mesa import Agent

from src.utils import (
    average_position,
    closest_agent,
    closest_position,
    grid_neighbours,
    manhattan_distance,
    move_away_from,
)


class Action(Enum):
    MOVE = "move"
    ATTACK = "attack"
    HEAL = "heal"
    SCAN = "scan"
    WAIT = "wait"


class SurvivorAgent(Agent):
    """Base class for all survivor agents."""

    def __init__(self, model, role_name: str, symbol: str):
        super().__init__(model)

        self.role_name = role_name
        self.symbol = symbol

        self.health = self.model.initial_health
        self.max_health = self.model.initial_health
        self.alive = True

        self.last_action: Optional[Action] = Action.WAIT
        self.reached_safe_zone = False
        self.visited_positions = set()

    def step(self):
        raise NotImplementedError

    # ==============================
    # Basic actions
    # ==============================

    def take_damage(self, amount: int):
        self.health -= amount

        if self.health <= 0:
            self.alive = False

            if self.pos is not None:
                self.model.grid.remove_agent(self)

            self.remove()

    def move_to(self, new_pos):
        if self.model.can_move_to(new_pos, moving_agent=self):
            self.model.grid.move_agent(self, new_pos)
            self.last_action = Action.MOVE
        else:
            self.last_action = Action.WAIT

    def attack_nearby_zombie(self) -> bool:
        for zombie in self.model.get_alive_zombies():
            if manhattan_distance(self.pos, zombie.pos) <= self.model.attack_range:
                zombie.health -= 1
                self.last_action = Action.ATTACK
                self.model.cooperation_events += 1

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
                if manhattan_distance(self.pos, survivor.pos) <= self.model.heal_range:
                    survivor.health = min(survivor.max_health, survivor.health + 1)
                    self.model.cooperation_events += 1
                    self.last_action = Action.HEAL
                    return True

        return False

    # ==============================
    # Safe zone helpers
    # ==============================

    def check_safe_zone(self):
        if self.pos in self.model.safe_zone_positions:
            self.reached_safe_zone = True

    def move_deeper_into_safe_zone(self):
        if self.pos not in self.model.safe_zone_positions:
            return

        current_distance = manhattan_distance(self.pos, self.model.safe_zone_pos)

        candidates = [
            pos for pos in grid_neighbours(self.pos)
            if pos in self.model.safe_zone_positions
            and self.model.can_move_to(pos, moving_agent=self)
        ]

        deeper_candidates = [
            pos for pos in candidates
            if manhattan_distance(pos, self.model.safe_zone_pos) < current_distance
        ]

        if not deeper_candidates:
            self.last_action = Action.WAIT
            return

        best_pos = closest_position(self.model.safe_zone_pos, deeper_candidates)
        self.move_to(best_pos)

    def stay_if_reached_safe_zone(self) -> bool:
        if self.pos in self.model.safe_zone_positions:
            self.reached_safe_zone = True
            self.move_deeper_into_safe_zone()
            return True

        return False

    def discover_safe_zone_if_visible(self) -> bool:
        visible_cells = [
            pos for pos in self.model.safe_zone_positions
            if manhattan_distance(self.pos, pos) <= self.model.vision_range
        ]

        return self.share_safe_zone_cells(visible_cells)

    def scout_scan(self) -> bool:
        scanned_cells = [
            pos for pos in self.model.safe_zone_positions
            if manhattan_distance(self.pos, pos) <= self.model.scout_scan_range
        ]

        self.last_action = Action.SCAN

        if not self.share_safe_zone_cells(scanned_cells):
            return False

        self.model.cooperation_events += 1
        return True

    def share_safe_zone_cells(self, cells) -> bool:
        if not cells:
            return False

        self.model.safe_zone_discovered = True

        for pos in cells:
            if pos not in self.model.discovered_safe_zone_positions:
                self.model.discovered_safe_zone_positions.append(pos)

        return True

    def get_team_goal(self):
        if not self.model.safe_zone_discovered:
            return None

        return closest_position(self.pos, self.model.discovered_safe_zone_positions)

    # ==============================
    # Perception helpers
    # ==============================

    def nearby_zombie(self, max_distance=None):
        if max_distance is None:
            max_distance = self.model.vision_range

        zombie = self.model.nearest_zombie(self.pos)

        if zombie is None:
            return None

        if manhattan_distance(self.pos, zombie.pos) <= max_distance:
            return zombie

        return None

    def teammate_in_danger(self, max_distance=None):
        if max_distance is None:
            max_distance = self.model.defender_intercept_range

        teammates = [s for s in self.model.get_alive_survivors() if s != self]
        zombies = self.model.get_alive_zombies()

        if not teammates or not zombies:
            return None

        threatened = [
            teammate for teammate in teammates
            if min(manhattan_distance(teammate.pos, zombie.pos) for zombie in zombies) <= max_distance
        ]

        return closest_agent(self.pos, threatened)

    def nearest_zombie_to_survivor(self, survivor):
        return closest_agent(survivor.pos, self.model.get_alive_zombies())

    def most_threatened_survivor(self):
        survivors = [s for s in self.model.get_alive_survivors() if s != self]
        zombies = self.model.get_alive_zombies()

        if not survivors or not zombies:
            return None

        def threat_score(survivor):
            closest_zombie_distance = min(
                manhattan_distance(survivor.pos, zombie.pos)
                for zombie in zombies
            )
            support_priority = -1 if survivor.role_name == "Support" else 0
            return (closest_zombie_distance, survivor.health, support_priority)

        return min(survivors, key=threat_score)

    def closest_injured_teammate(self):
        injured = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self and survivor.health < survivor.max_health
        ]

        return closest_agent(self.pos, injured)

    def team_is_too_far(self, max_distance):
        teammates = [s for s in self.model.get_alive_survivors() if s != self]

        if not teammates:
            return False

        return max(manhattan_distance(self.pos, teammate.pos) for teammate in teammates) > max_distance

    def get_survivor_by_role(self, role_name: str):
        for survivor in self.model.get_alive_survivors():
            if survivor.role_name == role_name:
                return survivor

        return None

    def get_team_center(self):
        positions = [survivor.pos for survivor in self.model.get_alive_survivors()]
        return average_position(positions, fallback=self.pos)

    def zombie_risk(self, pos):
        zombies = self.model.get_alive_zombies()

        if not zombies:
            return 99

        return min(manhattan_distance(pos, zombie.pos) for zombie in zombies)

    # ==============================
    # Movement helpers
    # ==============================

    def move_towards_position(self, target_pos, avoid_zombies=True, allow_wait=True):
        if target_pos is None:
            self.last_action = Action.WAIT
            return

        candidates = [
            pos for pos in grid_neighbours(self.pos)
            if self.model.can_move_to(pos, moving_agent=self)
        ]

        if allow_wait:
            candidates.append(self.pos)

        if not candidates:
            self.last_action = Action.WAIT
            return

        current_distance = manhattan_distance(self.pos, target_pos)
        progress_moves = [
            pos for pos in candidates
            if manhattan_distance(pos, target_pos) < current_distance
        ]

        if progress_moves:
            candidates = progress_moves
        elif allow_wait:
            self.last_action = Action.WAIT
            return

        def score(pos):
            danger_penalty = 0
            risk = self.zombie_risk(pos)

            if avoid_zombies and risk <= self.model.attack_range:
                danger_penalty = 30

            return (
                danger_penalty,
                manhattan_distance(pos, target_pos),
                -risk,
            )

        best_pos = min(candidates, key=score)

        if best_pos == self.pos:
            self.last_action = Action.WAIT
        else:
            self.move_to(best_pos)

    def move_near_position(self, target_pos, desired_distance=1, avoid_zombies=True):
        if target_pos is None:
            self.last_action = Action.WAIT
            return

        if manhattan_distance(self.pos, target_pos) <= desired_distance:
            self.last_action = Action.WAIT
            return

        self.move_towards_position(
            target_pos,
            avoid_zombies=avoid_zombies,
            allow_wait=False,
        )

    def explore(self):
        self.visited_positions.add(self.pos)

        candidates = [
            pos for pos in grid_neighbours(self.pos)
            if self.model.can_move_to(pos, moving_agent=self)
        ]

        if not candidates:
            self.last_action = Action.WAIT
            return

        unvisited = [pos for pos in candidates if pos not in self.visited_positions]
        if unvisited:
            candidates = unvisited

        safe = [pos for pos in candidates if self.zombie_risk(pos) > self.model.attack_range]
        if safe:
            candidates = safe

        team_center = self.get_team_center()
        spawn_reference = self.model.start_positions[0]

        def exploration_score(pos):
            return (
                manhattan_distance(pos, spawn_reference),
                manhattan_distance(pos, team_center),
                self.zombie_risk(pos),
            )

        best_score = max(exploration_score(pos) for pos in candidates)
        best_candidates = [pos for pos in candidates if exploration_score(pos) == best_score]
        chosen_pos = self.model.random.choice(best_candidates)

        self.move_to(chosen_pos)

    def move_away_towards_team(self, danger_pos):
        candidates = [
            pos for pos in grid_neighbours(self.pos)
            if self.model.can_move_to(pos, moving_agent=self)
        ]

        if not candidates:
            self.last_action = Action.WAIT
            return

        defender = self.get_survivor_by_role("Defender")
        team_target = defender.pos if defender is not None else self.get_team_center()
        current_danger_distance = manhattan_distance(self.pos, danger_pos)

        def score(pos):
            risk = self.zombie_risk(pos)
            escape_penalty = 20 if manhattan_distance(pos, danger_pos) <= current_danger_distance else 0
            zombie_penalty = 30 if risk <= self.model.attack_range else 0

            return (
                zombie_penalty,
                escape_penalty,
                manhattan_distance(pos, team_target),
                -risk,
            )

        self.move_to(min(candidates, key=score))



# ==============================
# Role-specific agents
# ==============================

class ScoutAgent(SurvivorAgent):
    """Explorer and information-discovery role."""

    def __init__(self, model):
        super().__init__(model, role_name="Scout", symbol="C")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.discover_safe_zone_if_visible()
        self.scout_scan()

        goal = self.get_team_goal()

        if goal is not None:
            zombie = self.nearby_zombie(max_distance=self.model.scout_escape_range)

            if zombie is not None:
                self.move_away_towards_team(zombie.pos)
            else:
                self.move_towards_position(goal, avoid_zombies=True, allow_wait=False)

            self.check_safe_zone()
            return

        zombie = self.nearby_zombie(max_distance=self.model.scout_escape_range)
        if zombie is not None:
            self.move_away_towards_team(zombie.pos)
            self.check_safe_zone()
            return

        threatened_teammate = self.teammate_in_danger(
            max_distance=self.model.defender_intercept_range
        )
        if threatened_teammate is not None:
            self.move_near_position(threatened_teammate.pos, desired_distance=1, avoid_zombies=True)
            self.check_safe_zone()
            return

        if self.team_is_too_far(self.model.scout_regroup_distance):
            self.move_towards_position(self.get_team_center(), avoid_zombies=True, allow_wait=False)
            self.check_safe_zone()
            return

        self.explore()
        self.check_safe_zone()


class DefenderAgent(SurvivorAgent):
    """Combat and protection role."""

    def __init__(self, model):
        super().__init__(model, role_name="Defender", symbol="D")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.discover_safe_zone_if_visible()

        if self.attack_nearby_zombie():
            self.check_safe_zone()
            return

        goal = self.get_team_goal()
        scout = self.get_survivor_by_role("Scout")
        support = self.get_survivor_by_role("Support")

        threatened = self.most_threatened_survivor()
        if threatened is not None:
            danger = self.nearest_zombie_to_survivor(threatened)

            if danger is not None:
                danger_to_teammate = manhattan_distance(danger.pos, threatened.pos)
                defender_to_danger = manhattan_distance(self.pos, danger.pos)

                if (
                    danger_to_teammate <= self.model.defender_intercept_range
                    or defender_to_danger <= self.model.defender_guard_range
                ):
                    self.move_towards_position(danger.pos, avoid_zombies=False, allow_wait=False)
                    self.check_safe_zone()
                    return

        if goal is not None:
            self.move_towards_position(goal, avoid_zombies=False, allow_wait=False)
            self.check_safe_zone()
            return

        injured = self.closest_injured_teammate()
        if injured is not None:
            self.move_near_position(injured.pos, desired_distance=1, avoid_zombies=False)
            self.check_safe_zone()
            return

        if scout is not None:
            self.move_near_position(
                scout.pos,
                desired_distance=self.model.defender_follow_scout_distance,
                avoid_zombies=False,
            )
            self.check_safe_zone()
            return

        if support is not None:
            if not self.model.safe_zone_discovered:
                self.explore()
            else:
                self.move_near_position(support.pos, desired_distance=1, avoid_zombies=False)

            self.check_safe_zone()
            return

        self.explore()
        self.check_safe_zone()


class SupportAgent(SurvivorAgent):
    """Healing and survival-support role."""

    def __init__(self, model):
        super().__init__(model, role_name="Support", symbol="S")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.discover_safe_zone_if_visible()

        if self.heal_nearby_survivor():
            self.check_safe_zone()
            return

        defender = self.get_survivor_by_role("Defender")
        scout = self.get_survivor_by_role("Scout")
        goal = self.get_team_goal()

        zombie = self.nearby_zombie(max_distance=self.model.support_escape_range)
        if zombie is not None:
            if defender is not None:
                self.move_near_position(defender.pos, desired_distance=1, avoid_zombies=True)
            else:
                escape_pos = move_away_from(self.pos, zombie.pos, rng=self.model.random)
                if self.model.can_move_to(escape_pos, moving_agent=self):
                    self.move_to(escape_pos)
                else:
                    self.last_action = Action.WAIT

            self.check_safe_zone()
            return

        injured = self.closest_injured_teammate()
        if injured is not None:
            if manhattan_distance(self.pos, injured.pos) <= self.model.support_help_injured_range:
                self.move_towards_position(injured.pos, avoid_zombies=True, allow_wait=False)
                self.check_safe_zone()
                return

        if goal is not None:
            self.move_towards_position(goal, avoid_zombies=False, allow_wait=False)
            self.check_safe_zone()
            return

        if defender is not None:
            self.move_near_position(
                defender.pos,
                desired_distance=self.model.support_follow_defender_distance,
                avoid_zombies=True,
            )
            self.check_safe_zone()
            return

        if scout is not None:
            self.move_near_position(
                scout.pos,
                desired_distance=self.model.support_follow_scout_distance,
                avoid_zombies=True,
            )
            self.check_safe_zone()
            return

        self.explore()
        self.check_safe_zone()
