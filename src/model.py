from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from config.config import INITIAL_HEALTH

from src.agents.environment_agents import ObstacleAgent, SafeZoneAgent
from src.agents.zombie_agent import ZombieAgent
from src.agents.survivor_agents import ScoutAgent, DefenderAgent, SupportAgent, AdaptiveAgent
from src.utils.utils import manhattan_distance

from src.metrics.metrics import (
    model_success,
    model_alive_survivors,
    model_cooperation_events,
    model_avg_distance,
)


class ZombieSurvivalModel(Model):
    """
    Mesa model for the zombie survival grid world.

    The environment contains:
    - survivor agents;
    - zombies;
    - obstacles;
    - one safe zone.
    """

    def __init__(
        self,
        *,
        width=12,
        height=12,
        num_zombies=4,
        num_obstacles=12,
        max_steps=80,
        team_mode="baseline",
        seed=None,
    ):
        super().__init__(rng=seed)

        self.width = width
        self.height = height

        self.num_zombies = num_zombies
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps

        self.team_mode = team_mode
        self.initial_health = INITIAL_HEALTH

        self.grid = MultiGrid(width, height, torus=False)

        self.safe_zone_pos = (width - 1, height - 1)
        self.safe_zone_agent = None

        # Reservar posições iniciais da equipa
        self.start_positions = [
            (0, 0),
            (0, 1),
            (1, 0),
        ]

        self.current_step = 0
        self.finished = False

        self.cooperation_events = 0

        self.create_safe_zone()
        self.create_obstacles()
        self.create_survivor_team()
        self.create_zombies()

        self.datacollector = DataCollector(
            model_reporters={
                "Success": model_success,
                "AliveSurvivors": model_alive_survivors,
                "CooperationEvents": model_cooperation_events,
                "AvgDistance": model_avg_distance,
            }
        )

    # ==============================
    # World creation
    # ==============================

    def create_safe_zone(self):
        safe_zone = SafeZoneAgent(self)
        self.grid.place_agent(safe_zone, self.safe_zone_pos)
        self.safe_zone_agent = safe_zone

    def create_obstacles(self):
        created = 0

        while created < self.num_obstacles:
            pos = self.random_empty_position(
                forbidden_positions=set(self.start_positions + [self.safe_zone_pos])
            )

            obstacle = ObstacleAgent(self)
            self.grid.place_agent(obstacle, pos)

            created += 1

    def create_survivor_team(self):
        if self.team_mode == "baseline":
            team = [
                ScoutAgent(self),
                DefenderAgent(self),
                SupportAgent(self),
            ]

        elif self.team_mode == "adaptive_replaces_scout":
            team = [
                AdaptiveAgent(self),
                DefenderAgent(self),
                SupportAgent(self),
            ]

        elif self.team_mode == "adaptive_replaces_defender":
            team = [
                ScoutAgent(self),
                AdaptiveAgent(self),
                SupportAgent(self),
            ]

        elif self.team_mode == "adaptive_replaces_support":
            team = [
                ScoutAgent(self),
                DefenderAgent(self),
                AdaptiveAgent(self),
            ]

        else:
            raise ValueError(f"Unknown team_mode: {self.team_mode}")

        for agent, pos in zip(team, self.start_positions):
            self.grid.place_agent(agent, pos)

    def create_zombies(self):
        created = 0

        while created < self.num_zombies:
            pos = self.random_empty_position(
                forbidden_positions=set(self.start_positions + [self.safe_zone_pos])
            )

            zombie = ZombieAgent(self)
            self.grid.place_agent(zombie, pos)

            created += 1

    def random_empty_position(self, forbidden_positions=None):
        if forbidden_positions is None:
            forbidden_positions = set()

        while True:
            pos = (
                self.random.randrange(self.width),
                self.random.randrange(self.height),
            )

            if pos in forbidden_positions:
                continue

            if self.is_cell_free_for_spawning(pos):
                return pos

    # ==============================
    # Entity access
    # ==============================

    def get_alive_survivors(self):
        return [
            agent for agent in self.agents
            if isinstance(agent, (ScoutAgent, DefenderAgent, SupportAgent, AdaptiveAgent))
            and agent.alive
        ]

    def get_alive_zombies(self):
        return [
            agent for agent in self.agents
            if isinstance(agent, ZombieAgent)
            and agent.alive
        ]

    # ==============================
    # Search helpers
    # ==============================

    def nearest_zombie(self, pos):
        zombies = self.get_alive_zombies()

        if not zombies:
            return None

        return min(
            zombies,
            key=lambda zombie: manhattan_distance(pos, zombie.pos)
        )

    def nearest_survivor(self, pos, exclude=None):
        survivors = [
            survivor for survivor in self.get_alive_survivors()
            if survivor != exclude
        ]

        if not survivors:
            return None

        return min(
            survivors,
            key=lambda survivor: manhattan_distance(pos, survivor.pos)
        )

    # ==============================
    # Movement and cell checks
    # ==============================

    def is_cell_free_for_spawning(self, pos):
        if self.grid.out_of_bounds(pos):
            return False

        contents = self.grid.get_cell_list_contents([pos])

        return len(contents) == 0

    def can_move_to(self, pos):
        """
        Agents can move to:
        - empty cells;
        - the safe zone cell.

        Agents cannot move to:
        - obstacles;
        - living zombies;
        - living survivors.
        """
        if self.grid.out_of_bounds(pos):
            return False

        contents = self.grid.get_cell_list_contents([pos])

        for obj in contents:
            if isinstance(obj, ObstacleAgent):
                return False

            if isinstance(obj, ZombieAgent) and obj.alive:
                return False

            if isinstance(obj, (ScoutAgent, DefenderAgent, SupportAgent, AdaptiveAgent)) and obj.alive:
                return False

        return True

    # ==============================
    # Mission status
    # ==============================

    def is_successful(self):
        return any(
            survivor.alive and survivor.pos == self.safe_zone_pos
            for survivor in self.get_alive_survivors()
        )

    def update_finished_status(self):
        if self.is_successful():
            self.finished = True
            return

        if len(self.get_alive_survivors()) == 0:
            self.finished = True
            return

        if self.current_step >= self.max_steps:
            self.finished = True
            return

    # ==============================
    # Simulation step
    # ==============================

    def step(self):
        if self.finished:
            return

        self.current_step += 1

        survivors = self.get_alive_survivors()
        zombies = self.get_alive_zombies()

        self.random.shuffle(survivors)
        self.random.shuffle(zombies)

        for survivor in survivors:
            survivor.step()

        for zombie in zombies:
            zombie.step()

        self.datacollector.collect(self)
        self.update_finished_status()