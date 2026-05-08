from src.model import ZombieSurvivalModel
from src.agents.environment_agents import ObstacleAgent, SafeZoneAgent
from config.config import GRID_HEIGHT, GRID_WIDTH, MAX_STEPS, NUM_OBSTACLES, NUM_ZOMBIES 


model = ZombieSurvivalModel(
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    num_zombies=NUM_ZOMBIES,
    num_obstacles=NUM_OBSTACLES,
    max_steps=MAX_STEPS,
    team_mode="baseline",
    seed=42,
)

print("Total agents:", len(model.agents))
print("Survivors:", len(model.get_alive_survivors()))
print("Zombies:", len(model.get_alive_zombies()))

print("Obstacles:", len([
    agent for agent in model.agents
    if isinstance(agent, ObstacleAgent)
]))

print("Safe zones:", len([
    agent for agent in model.agents
    if isinstance(agent, SafeZoneAgent)
]))

print("\nAll agents:")
for agent in model.agents:
    print(type(agent).__name__, agent.pos)