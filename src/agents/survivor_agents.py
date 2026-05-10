from enum import Enum
from typing import Optional, Dict

from mesa import Agent

from src.utils import manhattan_distance, move_away_from


class Action(Enum):
    MOVE = "move"
    ATTACK = "attack"
    HEAL = "heal"
    SCAN = "scan"
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

        # Used for exploration when the safe zone is unknown.
        self.visited_positions = set()

    def step(self):
        raise NotImplementedError

    # ======================================================
    # Basic actions
    # ======================================================

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
                    survivor.health = min(
                        survivor.max_health,
                        survivor.health + 1
                    )

                    self.model.cooperation_events += 1
                    self.last_action = Action.HEAL
                    return True

        return False

    # ======================================================
    # Safe zone discovery / goal helpers
    # ======================================================

    def check_safe_zone(self):
        if self.pos in self.model.safe_zone_positions:
            self.reached_safe_zone = True

    def move_deeper_into_safe_zone(self):
        """
        If the survivor is already inside the safe zone, it tries to move
        deeper into it so it does not block the entrance for teammates.
        """
        if self.pos not in self.model.safe_zone_positions:
            return

        current_distance = manhattan_distance(
            self.pos,
            self.model.safe_zone_pos
        )

        candidates = [
            pos for pos in self.neighbours(self.pos)
            if (
                pos in self.model.safe_zone_positions
                and self.model.can_move_to(pos, moving_agent=self)
            )
        ]

        if not candidates:
            self.last_action = Action.WAIT
            return

        deeper_candidates = [
            pos for pos in candidates
            if manhattan_distance(pos, self.model.safe_zone_pos) < current_distance
        ]

        if not deeper_candidates:
            self.last_action = Action.WAIT
            return

        best_pos = min(
            deeper_candidates,
            key=lambda pos: manhattan_distance(pos, self.model.safe_zone_pos)
        )

        self.move_to(best_pos)

    def stay_if_reached_safe_zone(self) -> bool:
        """
        If the survivor has reached the safe zone, it tries to move deeper
        into the safe zone before waiting.

        This prevents the first survivor from blocking the entrance.
        """
        if self.pos in self.model.safe_zone_positions:
            self.reached_safe_zone = True
            self.move_deeper_into_safe_zone()
            return True

        return False

    def visible_safe_zone_cells(self):
        """
        Returns safe-zone cells that are inside this agent's vision range.
        """
        visible_cells = []

        for pos in self.model.safe_zone_positions:
            if manhattan_distance(self.pos, pos) <= self.model.vision_range:
                visible_cells.append(pos)

        return visible_cells

    def discover_safe_zone_if_visible(self) -> bool:
        """
        If the agent sees the safe zone using normal vision, it shares this
        information with the team.
        """
        visible_cells = self.visible_safe_zone_cells()

        if not visible_cells:
            return False

        self.model.safe_zone_discovered = True

        for pos in visible_cells:
            if pos not in self.model.discovered_safe_zone_positions:
                self.model.discovered_safe_zone_positions.append(pos)

        return True

    def scout_scan(self) -> bool:
        """
        Special Scout ability.

        The Scout can scan farther than normal vision range.
        If the safe zone is within scout_scan_range, the Scout discovers it
        and shares the information with the team.

        This gives the Scout a unique role: information discovery.
        """
        scanned_cells = []

        for pos in self.model.safe_zone_positions:
            if manhattan_distance(self.pos, pos) <= self.model.scout_scan_range:
                scanned_cells.append(pos)

        self.last_action = Action.SCAN

        if not scanned_cells:
            return False

        self.model.safe_zone_discovered = True

        for pos in scanned_cells:
            if pos not in self.model.discovered_safe_zone_positions:
                self.model.discovered_safe_zone_positions.append(pos)

        self.model.cooperation_events += 1
        return True

    def get_closest_safe_zone_cell(self):
        """
        Returns the closest discovered safe-zone cell.
        If the safe zone has not been discovered yet, returns None.
        """
        if not self.model.safe_zone_discovered:
            return None

        if not self.model.discovered_safe_zone_positions:
            return None

        return min(
            self.model.discovered_safe_zone_positions,
            key=lambda pos: manhattan_distance(self.pos, pos)
        )

    def get_team_goal(self):
        return self.get_closest_safe_zone_cell()

    def call_team_to_safe_zone(self):
        """
        Scout updates the current leader position.
        """
        self.model.team_leader_pos = self.pos

    # ======================================================
    # Perception helpers
    # ======================================================

    def nearby_zombie(self, max_distance=None):
        if max_distance is None:
            max_distance = self.model.vision_range

        nearest_zombie = self.model.nearest_zombie(self.pos)

        if nearest_zombie is None:
            return None

        if manhattan_distance(self.pos, nearest_zombie.pos) <= max_distance:
            return nearest_zombie

        return None

    def closest_zombie(self):
        return self.model.nearest_zombie(self.pos)

    def teammate_in_danger(self, max_distance=None):
        """
        Returns a teammate that is currently close to a zombie.
        """
        if max_distance is None:
            max_distance = self.model.defender_intercept_range

        teammates = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self
        ]

        zombies = self.model.get_alive_zombies()

        if not teammates or not zombies:
            return None

        threatened_teammates = []

        for teammate in teammates:
            closest_zombie_distance = min(
                manhattan_distance(teammate.pos, zombie.pos)
                for zombie in zombies
            )

            if closest_zombie_distance <= max_distance:
                threatened_teammates.append(teammate)

        if not threatened_teammates:
            return None

        return min(
            threatened_teammates,
            key=lambda teammate: manhattan_distance(self.pos, teammate.pos)
        )

    def teammate_threatened_by_zombie(self, max_distance=None):
        """
        Returns a zombie threatening any alive teammate.
        """
        if max_distance is None:
            max_distance = self.model.defender_threat_range

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

    def nearest_zombie_to_survivor(self, survivor):
        zombies = self.model.get_alive_zombies()

        if not zombies:
            return None

        return min(
            zombies,
            key=lambda zombie: manhattan_distance(zombie.pos, survivor.pos)
        )

    def most_threatened_survivor(self):
        """
        Returns the teammate currently in most danger.
        Used by Defender.
        """
        survivors = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self
        ]

        zombies = self.model.get_alive_zombies()

        if not survivors or not zombies:
            return None

        def threat_score(survivor):
            closest_zombie_distance = min(
                manhattan_distance(survivor.pos, zombie.pos)
                for zombie in zombies
            )

            support_priority = -1 if survivor.role_name == "Support" else 0

            return (
                closest_zombie_distance,
                survivor.health,
                support_priority,
            )

        return min(survivors, key=threat_score)

    def closest_injured_teammate(self):
        injured = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self and survivor.health < survivor.max_health
        ]

        if not injured:
            return None

        return min(
            injured,
            key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
        )

    def team_is_too_far(self, max_distance):
        teammates = [
            survivor for survivor in self.model.get_alive_survivors()
            if survivor != self
        ]

        if not teammates:
            return False

        farthest_distance = max(
            manhattan_distance(self.pos, teammate.pos)
            for teammate in teammates
        )

        return farthest_distance > max_distance

    # ======================================================
    # Team helpers
    # ======================================================

    def get_survivor_by_role(self, role_name: str):
        for survivor in self.model.get_alive_survivors():
            if survivor.role_name == role_name:
                return survivor

        return None

    def neighbours(self, pos):
        x, y = pos

        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]

    def get_team_center(self):
        survivors = self.model.get_alive_survivors()

        if not survivors:
            return self.pos

        x = sum(survivor.pos[0] for survivor in survivors) // len(survivors)
        y = sum(survivor.pos[1] for survivor in survivors) // len(survivors)

        return (x, y)

    def zombie_risk(self, pos):
        zombies = self.model.get_alive_zombies()

        if not zombies:
            return 99

        return min(
            manhattan_distance(pos, zombie.pos)
            for zombie in zombies
        )

    # ======================================================
    # Movement helpers
    # ======================================================

    def move_towards_position(self, target_pos, avoid_zombies=True, allow_wait=True):
        """
        Move one cell toward target.

        If allow_wait is False, the agent may move sideways if no direct progress
        move exists. This helps avoid permanent waiting loops.
        """
        if target_pos is None:
            self.last_action = Action.WAIT
            return

        candidates = [
            pos for pos in self.neighbours(self.pos)
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
            distance_to_target = manhattan_distance(pos, target_pos)
            risk = self.zombie_risk(pos)

            danger_penalty = 0
            if avoid_zombies and risk <= self.model.attack_range:
                danger_penalty = 30

            return (
                danger_penalty,
                distance_to_target,
                -risk,
            )

        best_pos = min(candidates, key=score)

        if best_pos == self.pos:
            self.last_action = Action.WAIT
        else:
            self.move_to(best_pos)

    def move_near_position(self, target_pos, desired_distance=1, avoid_zombies=True):
        """
        Move near a target instead of trying to move onto the target cell.
        Useful for following teammates without blocking them.
        """
        if target_pos is None:
            self.last_action = Action.WAIT
            return

        current_distance = manhattan_distance(self.pos, target_pos)

        if current_distance <= desired_distance:
            self.last_action = Action.WAIT
            return

        self.move_towards_position(
            target_pos,
            avoid_zombies=avoid_zombies,
            allow_wait=False
        )

    def explore(self):
        """
        Exploration behaviour.

        The agent maps visited cells and gives priority to unvisited cells.
        It avoids immediately dangerous cells when possible.
        Among good options, it chooses randomly, but with a preference for
        moving away from already explored/team-start areas.
        """
        self.visited_positions.add(self.pos)

        candidates = [
            pos for pos in self.neighbours(self.pos)
            if self.model.can_move_to(pos, moving_agent=self)
        ]

        if not candidates:
            self.last_action = Action.WAIT
            return

        # 1. Prefer cells that were not visited before.
        unvisited_candidates = [
            pos for pos in candidates
            if pos not in self.visited_positions
        ]

        if unvisited_candidates:
            candidates = unvisited_candidates

        # 2. Avoid cells adjacent to zombies if possible.
        safe_candidates = [
            pos for pos in candidates
            if self.zombie_risk(pos) > self.model.attack_range
        ]

        if safe_candidates:
            candidates = safe_candidates

        # 3. Give a small exploration bias:
        # prefer cells farther from the team center and farther from spawn.
        team_center = self.get_team_center()
        spawn_reference = self.model.start_positions[0]

        def exploration_score(pos):
            distance_from_team = manhattan_distance(pos, team_center)
            distance_from_spawn = manhattan_distance(pos, spawn_reference)
            risk = self.zombie_risk(pos)

            return (
                distance_from_spawn,
                distance_from_team,
                risk,
            )

        # 4. Keep only the best-scoring candidates.
        best_score = max(
            exploration_score(pos)
            for pos in candidates
        )

        best_candidates = [
            pos for pos in candidates
            if exploration_score(pos) == best_score
        ]

        # 5. Random among the best candidates.
        chosen_pos = self.model.random.choice(best_candidates)

        self.move_to(chosen_pos)

    def move_away_towards_team(self, danger_pos):
        """
        Escape movement used mainly by the Scout.

        The agent tries to move away from danger but also tries to stay
        close to the Defender/team.
        """
        candidates = [
            pos for pos in self.neighbours(self.pos)
            if self.model.can_move_to(pos, moving_agent=self)
        ]

        if not candidates:
            self.last_action = Action.WAIT
            return

        defender = self.get_survivor_by_role("Defender")
        team_center = self.get_team_center()

        if defender is not None:
            team_target = defender.pos
        else:
            team_target = team_center

        current_danger_distance = manhattan_distance(self.pos, danger_pos)

        def score(pos):
            danger_distance = manhattan_distance(pos, danger_pos)
            team_distance = manhattan_distance(pos, team_target)
            risk = self.zombie_risk(pos)

            escape_penalty = 0
            if danger_distance <= current_danger_distance:
                escape_penalty = 20

            zombie_penalty = 0
            if risk <= self.model.attack_range:
                zombie_penalty = 30

            return (
                zombie_penalty,
                escape_penalty,
                team_distance,
                -risk,
            )

        best_pos = min(candidates, key=score)
        self.move_to(best_pos)

    def follow_team_goal(self):
        """
        Follow discovered safe zone if known.
        If it is unknown, follow the Scout.
        If there is no Scout, explore.
        """
        self.discover_safe_zone_if_visible()

        goal = self.get_team_goal()

        if goal is not None:
            self.move_towards_position(
                goal,
                avoid_zombies=True,
                allow_wait=False
            )

            self.check_safe_zone()
            return

        scout = self.get_survivor_by_role("Scout")

        if scout is not None and scout != self:
            self.move_near_position(
                scout.pos,
                desired_distance=1,
                avoid_zombies=True
            )

            self.check_safe_zone()
            return

        self.explore()
        self.check_safe_zone()


class ScoutAgent(SurvivorAgent):
    """
    Scout:
    - explores while the safe zone is unknown;
    - has a special scan ability to detect the safe zone from farther away;
    - discovers and shares the safe-zone location;
    - avoids zombies when they are close;
    - does not abandon the team.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Scout", symbol="C")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.call_team_to_safe_zone()

        # Normal vision discovery.
        self.discover_safe_zone_if_visible()

        # Scout special ability: scan farther than normal agents.
        # This makes Scout useful as an information role.
        self.scout_scan()

        goal = self.get_team_goal()

        # 1. If safe zone is known, lead the team to it.
        if goal is not None:
            nearest_zombie = self.nearby_zombie(max_distance=self.model.scout_escape_range)

            if nearest_zombie is not None:
                self.move_away_towards_team(nearest_zombie.pos)
            else:
                self.move_towards_position(
                    goal,
                    avoid_zombies=True,
                    allow_wait=False
                )

            self.check_safe_zone()
            return

        # 2. Immediate survival: if zombie is very close, retreat toward the team.
        nearest_zombie = self.nearby_zombie(max_distance=self.model.scout_escape_range)

        if nearest_zombie is not None:
            self.move_away_towards_team(nearest_zombie.pos)
            self.check_safe_zone()
            return

        # 3. If a teammate is in real danger, regroup.
        threatened_teammate = self.teammate_in_danger(
            max_distance=self.model.defender_intercept_range
        )

        if threatened_teammate is not None:
            self.move_near_position(
                threatened_teammate.pos,
                desired_distance=1,
                avoid_zombies=True
            )

            self.check_safe_zone()
            return

        # 4. If team is too spread out, regroup.
        if self.team_is_too_far(self.model.scout_regroup_distance):
            self.move_towards_position(
                self.get_team_center(),
                avoid_zombies=True,
                allow_wait=False
            )

            self.check_safe_zone()
            return

        # 5. Otherwise, continue exploration.
        self.explore()
        self.check_safe_zone()


class DefenderAgent(SurvivorAgent):
    """
    Defender:
    - attacks adjacent zombies;
    - intercepts zombies that are threatening teammates;
    - protects injured teammates when there is no immediate zombie target;
    - otherwise stays close to Scout.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Defender", symbol="D")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.discover_safe_zone_if_visible()

        # 1. Attack immediately if possible.
        attacked = self.attack_nearby_zombie()

        if attacked:
            self.check_safe_zone()
            return

        goal = self.get_team_goal()

        scout = self.get_survivor_by_role("Scout")
        support = self.get_survivor_by_role("Support")

        # 2. Protect the most threatened teammate, but only if the threat is relevant.
        threatened = self.most_threatened_survivor()

        if threatened is not None:
            danger = self.nearest_zombie_to_survivor(threatened)

            if danger is not None:
                distance_danger_to_teammate = manhattan_distance(
                    danger.pos,
                    threatened.pos
                )

                distance_defender_to_danger = manhattan_distance(
                    self.pos,
                    danger.pos
                )

                if (
                    distance_danger_to_teammate <= self.model.defender_intercept_range
                    or distance_defender_to_danger <= self.model.defender_guard_range
                ):
                    self.move_towards_position(
                        danger.pos,
                        avoid_zombies=False,
                        allow_wait=False
                    )

                    self.check_safe_zone()
                    return

        # 3. If safe zone is known, escort toward it.
        if goal is not None:
            self.move_towards_position(
                goal,
                avoid_zombies=False,
                allow_wait=False
            )

            self.check_safe_zone()
            return

        # 4. If someone is injured and there is no immediate zombie target,
        # stay close to protect them.
        injured = self.closest_injured_teammate()

        if injured is not None:
            self.move_near_position(
                injured.pos,
                desired_distance=1,
                avoid_zombies=False
            )

            self.check_safe_zone()
            return

        # 5. Normal behaviour: bodyguard the Scout.
        if scout is not None:
            self.move_near_position(
                scout.pos,
                desired_distance=self.model.defender_follow_scout_distance,
                avoid_zombies=False
            )

            self.check_safe_zone()
            return

        # 6. Fallback: stay close to Support if Scout is unavailable.
        if support is not None:
            self.move_near_position(
                support.pos,
                desired_distance=1,
                avoid_zombies=False
            )

            self.check_safe_zone()
            return

        self.explore()
        self.check_safe_zone()


class SupportAgent(SurvivorAgent):
    """
    Support:
    - heals nearby injured teammates;
    - avoids combat;
    - follows Defender/Scout;
    - supports extraction after safe zone discovery.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Support", symbol="S")

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.discover_safe_zone_if_visible()

        # 1. Heal adjacent teammate if possible.
        healed = self.heal_nearby_survivor()

        if healed:
            self.check_safe_zone()
            return

        defender = self.get_survivor_by_role("Defender")
        scout = self.get_survivor_by_role("Scout")
        goal = self.get_team_goal()

        # 2. Immediate survival: if zombie is adjacent, move away / toward Defender.
        nearest_zombie = self.nearby_zombie(max_distance=self.model.support_escape_range)

        if nearest_zombie is not None:
            if defender is not None:
                self.move_near_position(
                    defender.pos,
                    desired_distance=1,
                    avoid_zombies=True
                )
            else:
                escape_pos = move_away_from(
                    self.pos,
                    nearest_zombie.pos,
                    rng=self.model.random
                )

                if self.model.can_move_to(escape_pos, moving_agent=self):
                    self.move_to(escape_pos)
                else:
                    self.last_action = Action.WAIT

            self.check_safe_zone()
            return

        # 3. If there is an injured teammate nearby, go help.
        injured = self.closest_injured_teammate()

        if injured is not None:
            distance_to_injured = manhattan_distance(self.pos, injured.pos)

            if distance_to_injured <= self.model.support_help_injured_range:
                self.move_towards_position(
                    injured.pos,
                    avoid_zombies=True,
                    allow_wait=False
                )

                self.check_safe_zone()
                return

        # 4. If safe zone is known, move toward it.
        if goal is not None:
            self.move_towards_position(
                goal,
                avoid_zombies=False,
                allow_wait=False
            )

            self.check_safe_zone()
            return

        # 5. Normal formation: stay near Defender.
        if defender is not None:
            self.move_near_position(
                defender.pos,
                desired_distance=self.model.support_follow_defender_distance,
                avoid_zombies=True
            )

            self.check_safe_zone()
            return

        # 6. If Defender is dead/unavailable, stay near Scout.
        if scout is not None:
            self.move_near_position(
                scout.pos,
                desired_distance=self.model.support_follow_scout_distance,
                avoid_zombies=True
            )

            self.check_safe_zone()
            return

        self.explore()
        self.check_safe_zone()


class AdaptiveAgent(SurvivorAgent):
    """
    Adaptive ad hoc agent.

    Kept here so the existing team modes do not break.
    You can ignore this for now while testing the baseline.
    """

    def __init__(self, model):
        super().__init__(model, role_name="Adaptive", symbol="A")

        self.teammate_scores: Dict[int, Dict[str, int]] = {}

    def step(self):
        if not self.alive or self.model.finished or self.stay_if_reached_safe_zone():
            return

        self.discover_safe_zone_if_visible()

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

    # ======================================================
    # Observation / role inference
    # ======================================================

    def observe_teammates(self):
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

            if teammate.last_action == Action.SCAN:
                scores["scout"] += 2

            goal = teammate.get_team_goal()

            if goal is not None:
                distance_to_safe_zone = manhattan_distance(
                    teammate.pos,
                    goal
                )

                if distance_to_safe_zone <= self.model.vision_range:
                    scores["scout"] += 1

            nearest_zombie = self.model.nearest_zombie(teammate.pos)

            if nearest_zombie is not None:
                distance_to_zombie = manhattan_distance(
                    teammate.pos,
                    nearest_zombie.pos
                )

                if (
                    teammate.last_action == Action.MOVE
                    and distance_to_zombie > self.model.scout_escape_range
                ):
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

    # ======================================================
    # Adaptive role behaviours
    # ======================================================

    def scout_behaviour(self):
        self.call_team_to_safe_zone()

        self.discover_safe_zone_if_visible()

        # If Adaptive decides to act as Scout, it also uses the scan ability.
        self.scout_scan()

        goal = self.get_team_goal()

        nearest_zombie = self.nearby_zombie(max_distance=self.model.scout_escape_range)

        if nearest_zombie is not None:
            self.move_away_towards_team(nearest_zombie.pos)
            return

        if goal is not None:
            self.move_towards_position(
                goal,
                avoid_zombies=True,
                allow_wait=False
            )
            return

        self.explore()

    def defender_behaviour(self):
        attacked = self.attack_nearby_zombie()

        if attacked:
            return

        threatening_zombie = self.teammate_threatened_by_zombie(
            max_distance=self.model.defender_intercept_range
        )

        if threatening_zombie is not None:
            self.move_towards_position(
                threatening_zombie.pos,
                avoid_zombies=False,
                allow_wait=False
            )
            return

        self.follow_team_goal()

    def support_behaviour(self):
        healed = self.heal_nearby_survivor()

        if healed:
            return

        injured = self.closest_injured_teammate()

        if injured is not None:
            self.move_towards_position(
                injured.pos,
                avoid_zombies=True,
                allow_wait=False
            )
            return

        self.follow_team_goal()