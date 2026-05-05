from mesa import Agent

from src.utils.utils import manhattan_distance, move_towards


class ZombieAgent(Agent):
    """
    Zombie behaviour:
    - finds the closest alive survivor;
    - attacks if adjacent;
    - otherwise moves toward the closest survivor.
    """

    def __init__(self, model):
        super().__init__(model)

        self.health = 2
        self.alive = True

    def step(self):
        if not self.alive or self.model.finished:
            return

        survivors = self.model.get_alive_survivors()

        if not survivors:
            return

        target = min(
            survivors,
            key=lambda survivor: manhattan_distance(self.pos, survivor.pos)
        )

        distance = manhattan_distance(self.pos, target.pos)

        if distance == 1:
            target.take_damage(1)
            return

        new_pos = move_towards(self.pos, target.pos)

        if self.model.can_move_to(new_pos):
            self.model.grid.move_agent(self, new_pos)