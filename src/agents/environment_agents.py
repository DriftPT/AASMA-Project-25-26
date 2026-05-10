from mesa import Agent


class ObstacleAgent(Agent):
    """
    Static obstacle in the grid.
    Survivor agents and zombies cannot move through obstacles.
    """

    def __init__(self, model):
        super().__init__(model)


class SafeZoneAgent(Agent):
    """
    Target cell that survivor agents try to reach.
    The mission succeeds if all survivor agents reach this cells.
    """

    def __init__(self, model):
        super().__init__(model)