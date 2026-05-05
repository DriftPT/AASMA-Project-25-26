import time
import matplotlib.pyplot as plt
import solara

from src.model import ZombieSurvivalModel

from src.agents.survivor_agents import ScoutAgent, DefenderAgent, SupportAgent, AdaptiveAgent
from src.agents.zombie_agent import ZombieAgent
from src.agents.environment_agents import ObstacleAgent, SafeZoneAgent


# ==============================
# GLOBAL REACTIVE STATE
# ==============================

team_mode = solara.reactive("baseline")

refresh_counter = solara.reactive(0)

is_playing = solara.reactive(False)

model_state = solara.reactive(
    ZombieSurvivalModel(
        width=12,
        height=12,
        num_zombies=4,
        num_obstacles=12,
        max_steps=80,
        team_mode="baseline",
        seed=42,
    )
)


# ==============================
# MODEL CONTROL FUNCTIONS
# ==============================

def force_refresh():
    """
    Forces Solara to re-render the page.
    The model is modified in-place, so we need this counter to update the UI.
    """
    refresh_counter.value += 1


def reset_model():
    is_playing.value = False
    model_state.value = ZombieSurvivalModel(
        width=12,
        height=12,
        num_zombies=4,
        num_obstacles=12,
        max_steps=80,
        team_mode=team_mode.value,
        seed=42,
    )

    force_refresh()


def step_model():
    model = model_state.value

    if not model.finished:
        model.step()

    force_refresh()


def play_model():
    is_playing.value = True


def pause_model():
    is_playing.value = False


# ==============================
# DRAWING FUNCTION
# ==============================

def draw_grid(model):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(-0.5, model.width - 0.5)
    ax.set_ylim(-0.5, model.height - 0.5)

    ax.set_xticks(range(model.width))
    ax.set_yticks(range(model.height))

    ax.grid(True, linestyle="--", linewidth=0.5)

    ax.set_aspect("equal")
    ax.set_title("Zombie Survival Grid World")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for agent in list(model.agents):
        if getattr(agent, "pos", None) is None:
            continue

        x, y = agent.pos

        if isinstance(agent, ScoutAgent):
            ax.scatter(x, y, s=300, marker="o", color="blue")
            ax.text(x, y, "C", ha="center", va="center", color="white", weight="bold")

        elif isinstance(agent, DefenderAgent):
            ax.scatter(x, y, s=300, marker="s", color="blue")
            ax.text(x, y, "D", ha="center", va="center", color="white", weight="bold")

        elif isinstance(agent, SupportAgent):
            ax.scatter(x, y, s=300, marker="D", color="blue")
            ax.text(x, y, "S", ha="center", va="center", color="white", weight="bold")

        elif isinstance(agent, AdaptiveAgent):
            ax.scatter(x, y, s=450, marker="*", color="orange")
            ax.text(x, y, "A", ha="center", va="center", color="black", weight="bold")

        elif isinstance(agent, ZombieAgent):
            ax.scatter(x, y, s=350, marker="X", color="green")
            ax.text(x, y, "Z", ha="center", va="center", color="white", weight="bold")

        elif isinstance(agent, ObstacleAgent):
            ax.scatter(x, y, s=350, marker="s", color="black")

        elif isinstance(agent, SafeZoneAgent):
            ax.scatter(x, y, s=500, marker="P", color="gold")
            ax.text(x, y, "S", ha="center", va="center", color="black", weight="bold")

    return fig


# ==============================
# SOLARA PAGE
# ==============================

@solara.component
def Page():
    # This line forces the page to update when refresh_counter changes.
    _ = refresh_counter.value

    model = model_state.value

    def play_loop(cancel_event):
        while not cancel_event.is_set():
            if is_playing.value and not model_state.value.finished:
                model_state.value.step()
                force_refresh()

            if model_state.value.finished:
                is_playing.value = False

            time.sleep(0.7)

    solara.use_thread(play_loop, dependencies=[])

    with solara.Sidebar():
        solara.Markdown("## Controls")

        solara.Select(
            label="Team mode",
            values=[
                "baseline",
                "adaptive_replaces_scout",
                "adaptive_replaces_defender",
                "adaptive_replaces_support",
            ],
            value=team_mode,
        )

        solara.Button("Reset", on_click=reset_model)
        solara.Button("Step", on_click=step_model)
        solara.Button("Play", on_click=play_model)
        solara.Button("Pause", on_click=pause_model)

        solara.Markdown("## Information")
        solara.Markdown(f"**Step:** {model.current_step}")
        solara.Markdown(f"**Finished:** {model.finished}")
        solara.Markdown(f"**Success:** {model.is_successful()}")
        solara.Markdown(f"**Alive survivors:** {len(model.get_alive_survivors())}")
        solara.Markdown(f"**Alive zombies:** {len(model.get_alive_zombies())}")
        solara.Markdown(f"**Cooperation events:** {model.cooperation_events}")

        solara.Markdown("## Legend")
        solara.Markdown("""
- **C**: Scout  
- **D**: Defender  
- **S**: Support  
- **A**: Adaptive Agent  
- **Z**: Zombie  
- **Black square**: Obstacle  
- **S**: Safe Zone  
""")

    solara.Markdown("# Ad Hoc Teamwork in a Zombie Survival Grid World")

    fig = draw_grid(model)
    solara.FigureMatplotlib(fig)
    plt.close(fig)