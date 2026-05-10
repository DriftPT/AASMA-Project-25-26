from typing import Dict, Tuple
from src.agents.survivor_agents import SurvivorAgent, Action
from src.utils import manhattan_distance


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
        self.check_safe_zone()