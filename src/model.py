from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from config.config import INITIAL_HEALTH

from src.agents.environment_agents import ObstacleAgent, SafeZoneAgent
from src.agents.zombie_agent import ZombieAgent
from src.agents.survivor_agents import ScoutAgent, DefenderAgent, SupportAgent
from src.agents.adaptive_agent import AdaptiveAgent
from src.utils import manhattan_distance

from src.metrics import (
    model_success,
    model_alive_survivors,
    model_cooperation_events,
    model_avg_distance,
)

from config.config import GRID_WIDTH, GRID_HEIGHT, NUM_ZOMBIES, NUM_OBSTACLES, RANDOM_SEED, MAX_STEPS


class ZombieSurvivalModel(Model):
    """
    Mesa model for the zombie survival grid world.

    The safe zone exists from the beginning, but survivors do not know its
    location until it is discovered through vision or Scout scan.
    """

    def __init__(
        self,
        *,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode="baseline",
        seed=RANDOM_SEED,
        adaptive_policy=None,
    ):
        super().__init__(rng=seed)

        self.width = width
        self.height = height
        self.num_zombies = num_zombies
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.team_mode = team_mode
        self.adaptive_policy = adaptive_policy
        self.initial_health = INITIAL_HEALTH

        # ==============================
        # Agent perception / behaviour ranges
        # ==============================
        self.zombie_vision_range = 6
        self.vision_range = 5

        self.attack_range = 1
        self.heal_range = 1

        # Scout
        self.scout_escape_range = 1
        self.scout_regroup_distance = 6
        self.scout_scan_range = 9

        # Defender
        self.defender_intercept_range = 4
        self.defender_guard_range = 3
        self.defender_follow_scout_distance = 2

        # Support
        self.support_escape_range = 1
        self.support_follow_defender_distance = 1
        self.support_follow_scout_distance = 3
        self.support_help_injured_range = 4

        # ==============================
        # Shared discovered information
        # ==============================

        self.safe_zone_discovered = False
        self.discovered_safe_zone_positions = []

        self.grid = MultiGrid(width, height, torus=False)

        # ==============================
        # Safe zone and initial positions
        # ==============================

        self.safe_zone_positions = [
            (width - 2, height - 2),
            (width - 1, height - 2),
            (width - 2, height - 1),
            (width - 1, height - 1),
        ]
        self.safe_zone_pos = (width - 1, height - 1)
        self.safe_zone_agents = []

        self.start_positions = [
            (1, 1),  # Scout
            (0, 1),  # Defender
            (1, 0),  # Support
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
        for pos in self.safe_zone_positions:
            safe_zone = SafeZoneAgent(self)
            self.grid.place_agent(safe_zone, pos)
            self.safe_zone_agents.append(safe_zone)

    def create_obstacles(self):
        created = 0

        while created < self.num_obstacles:
            pos = self.random_empty_position(
                forbidden_positions=set(self.start_positions + self.safe_zone_positions)
            )

            obstacle = ObstacleAgent(self)
            self.grid.place_agent(obstacle, pos)
            created += 1

    def create_survivor_team(self):
        if self.team_mode == "baseline":
            team = [ScoutAgent(self), DefenderAgent(self), SupportAgent(self)]
        elif self.team_mode == "adaptive_replaces_scout":
            team = [AdaptiveAgent(self), DefenderAgent(self), SupportAgent(self)]
        elif self.team_mode == "adaptive_replaces_defender":
            team = [ScoutAgent(self), AdaptiveAgent(self), SupportAgent(self)]
        elif self.team_mode == "adaptive_replaces_support":
            team = [ScoutAgent(self), DefenderAgent(self), AdaptiveAgent(self)]
        elif self.team_mode == "adaptive_adaptive_adaptive":
            team = [AdaptiveAgent(self), AdaptiveAgent(self), AdaptiveAgent(self)]
        else:
            raise ValueError(f"Unknown team_mode: {self.team_mode}")

        for agent, pos in zip(team, self.start_positions):
            self.grid.place_agent(agent, pos)

    def create_zombies(self):
        created = 0

        while created < self.num_zombies:
            pos = self.random_empty_position(
                forbidden_positions=set(self.start_positions + self.safe_zone_positions)
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
            if isinstance(agent, ZombieAgent) and agent.alive
        ]

    def get_active_survivors(self):
        return [
            survivor for survivor in self.get_alive_survivors()
            if survivor.pos not in self.safe_zone_positions
        ]

    # ==============================
    # Search helpers
    # ==============================

    def nearest_zombie(self, pos):
        zombies = self.get_alive_zombies()

        if not zombies:
            return None

        return min(zombies, key=lambda zombie: manhattan_distance(pos, zombie.pos))

    # ==============================
    # Movement and cell checks
    # ==============================

    def is_cell_free_for_spawning(self, pos):
        if self.grid.out_of_bounds(pos):
            return False

        contents = self.grid.get_cell_list_contents([pos])
        return len(contents) == 0

    def can_move_to(self, pos, moving_agent=None):
        if self.grid.out_of_bounds(pos):
            return False

        if isinstance(moving_agent, ZombieAgent) and pos in self.safe_zone_positions:
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
        alive_survivors = self.get_alive_survivors()

        if len(alive_survivors) == 0:
            return False

        return all(
            survivor.pos in self.safe_zone_positions
            for survivor in alive_survivors
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
