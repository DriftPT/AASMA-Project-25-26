import os
import time
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import solara

from src.model import ZombieSurvivalModel

from src.agents.survivor_agents import ScoutAgent, DefenderAgent, SupportAgent, AdaptiveAgent
from src.agents.zombie_agent import ZombieAgent
from src.agents.environment_agents import ObstacleAgent, SafeZoneAgent
from config.config import GRID_HEIGHT, GRID_WIDTH, MAX_STEPS, NUM_OBSTACLES, NUM_ZOMBIES, RANDOM_SEED

# ==============================
# IMAGE ASSETS
# ===================

ASSETS_DIR = solara.Path(__file__).parent / "images"

IMAGE_PATHS = {
    "scout": ASSETS_DIR / "scout.png",
    "defender": ASSETS_DIR / "defender.png",
    "support": ASSETS_DIR / "support.png",
    "adaptive": ASSETS_DIR / "adaptive.png",
    "zombie": ASSETS_DIR / "zombie.png",
    "obstacle": ASSETS_DIR / "obstacle.png",
    "safezone": ASSETS_DIR / "safezone.png",
    "background": ASSETS_DIR / "background.png",
}

def draw_image(ax, image_path, x, y, zoom=0.12):
    """
    Draws an image centered on a grid position.
    If the image does not exist, it does nothing.
    """
    if not image_path.exists():
        return

    img = mpimg.imread(image_path)
    image = OffsetImage(img, zoom=zoom)
    box = AnnotationBbox(image, (x, y), frameon=False)
    ax.add_artist(box)

# ==============================
# GLOBAL REACTIVE STATE
# ==============================

team_mode = solara.reactive("baseline")

refresh_counter = solara.reactive(0)

is_playing = solara.reactive(False)

model_state = solara.reactive(
    ZombieSurvivalModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode="baseline",
        seed=RANDOM_SEED,
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
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode=team_mode.value,
        seed=RANDOM_SEED,
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

def shutdown_app():
    is_playing.value = False
    plt.close("all")
    os._exit(0)

# ==============================
# DRAWING FUNCTION
# ==============================

def draw_grid(model):
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(-0.5, model.width - 0.5)
    ax.set_ylim(-0.5, model.height - 0.5)

    background_img = mpimg.imread(IMAGE_PATHS["background"])

    ax.imshow(
        background_img,
        aspect="auto",
        zorder=0
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_frame_on(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("black")

    ax.set_aspect("equal")
    ax.set_title("Zombie Survival Grid World")

    for agent in list(model.agents):
        if getattr(agent, "pos", None) is None:
            continue

        x, y = agent.pos

        if isinstance(agent, ScoutAgent):
            draw_image(ax, IMAGE_PATHS["scout"], x, y, zoom=0.045)

        elif isinstance(agent, DefenderAgent):
            draw_image(ax, IMAGE_PATHS["defender"], x, y, zoom=0.045)

        elif isinstance(agent, SupportAgent):
            draw_image(ax, IMAGE_PATHS["support"], x, y, zoom=0.045)

        elif isinstance(agent, AdaptiveAgent):
            draw_image(ax, IMAGE_PATHS["adaptive"], x, y, zoom=0.045)

        elif isinstance(agent, ZombieAgent):
            draw_image(ax, IMAGE_PATHS["zombie"], x, y, zoom=0.045)

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

            time.sleep(0.5)

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
        solara.Button("Close App", on_click=shutdown_app, color="error")

        solara.Markdown("## Information")
        solara.Markdown(f"**Step:** {model.current_step}")
        solara.Markdown(f"**Finished:** {model.finished}")
        solara.Markdown(f"**Success:** {model.is_successful()}")
        solara.Markdown(f"**Alive survivors:** {len(model.get_alive_survivors())}")
        solara.Markdown(f"**Alive zombies:** {len(model.get_alive_zombies())}")
        solara.Markdown(f"**Cooperation events:** {model.cooperation_events}")

    solara.Markdown("# Ad Hoc Teamwork in a Zombie Survival Grid World")

    fig = draw_grid(model)
    solara.FigureMatplotlib(fig)
    plt.close(fig)