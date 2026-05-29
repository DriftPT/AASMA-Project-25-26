import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import solara

from pathlib import Path
from src.model import ZombieSurvivalModel
from src.experiments import train_adaptive_agent

from src.agents.survivor_agents import ScoutAgent, DefenderAgent, SupportAgent
from src.agents.adaptive_agent import AdaptiveAgent
from src.agents.zombie_agent import ZombieAgent
from src.agents.environment_agents import ObstacleAgent, SafeZoneAgent
from config.config import GRID_HEIGHT, GRID_WIDTH, MAX_STEPS, NUM_OBSTACLES, NUM_ZOMBIES, RANDOM_SEED

# ==============================
# IMAGE ASSETS
# ==============================

ASSETS_DIR = Path(__file__).parent / "images"

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

IMAGE_CACHE = {}

def load_image(name):
    if name not in IMAGE_CACHE:
        image_path = IMAGE_PATHS[name]

        if not image_path.exists():
            IMAGE_CACHE[name] = None
        else:
            IMAGE_CACHE[name] = mpimg.imread(image_path)

    return IMAGE_CACHE[name]

def draw_image(ax, image_name, x, y, zoom=0.12):
    """
    Draws a cached image centered on a grid position.
    The image is loaded only once to avoid memory problems.
    """
    img = load_image(image_name)

    if img is None:
        return

    image = OffsetImage(img, zoom=zoom)
    box = AnnotationBbox(image, (x, y), frameon=False)
    ax.add_artist(box)

# ==============================
# PROPORTIONAL SIZING HELPERS
# ==============================
 
# Reference grid for which the original zoom/sizes were tuned.
_REF_GRID = 20
 
def _scale(grid_size: int) -> float:
    """Linear scale factor relative to the reference 20x20 grid."""
    return _REF_GRID / grid_size
 
def image_zoom(grid_size: int) -> float:
    """Zoom for OffsetImage — shrinks proportionally as the grid grows."""
    return 0.045 * _scale(grid_size)
 
def scatter_size(grid_size: int, base: float) -> float:
    """Marker size for ax.scatter — area scales as scale^2."""
    return base * (_scale(grid_size) ** 2)
 
def health_bar_dims(grid_size: int):
    """Returns (bar_width, bar_height, y_offset) scaled to the grid."""
    s = _scale(grid_size)
    return 0.65 * s, 0.08 * s, 0.38 * s
 
def font_size(grid_size: int, base: float = 8.0) -> float:
    """Font size for text labels on the grid."""
    return max(4.0, base * _scale(grid_size))
 
 
def draw_image(ax, image_name, x, y, zoom=0.045):
    img = load_image(image_name)
    if img is None:
        return
    image = OffsetImage(img, zoom=zoom)
    box = AnnotationBbox(image, (x, y), frameon=False)
    ax.add_artist(box)

# ==============================
# GLOBAL REACTIVE STATE
# ==============================

team_mode = solara.reactive("baseline")

refresh_counter = solara.reactive(0)

is_playing = solara.reactive(False)

trained_adaptive_policies = None

model_state = solara.reactive(
    ZombieSurvivalModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode="baseline",
        seed=RANDOM_SEED,
        adaptive_policy=None,
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


def get_adaptive_policy(selected_team_mode):
    global trained_adaptive_policies
    
    if selected_team_mode == "baseline":
        return None

    if trained_adaptive_policies is None:
        trained_adaptive_policies = train_adaptive_agent()

    return trained_adaptive_policies

def reset_model():
    is_playing.value = False

    adaptive_policy = get_adaptive_policy(team_mode.value)

    model_state.value = ZombieSurvivalModel(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        num_zombies=NUM_ZOMBIES,
        num_obstacles=NUM_OBSTACLES,
        max_steps=MAX_STEPS,
        team_mode=team_mode.value,
        seed=RANDOM_SEED,
        adaptive_policy=adaptive_policy,
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
def draw_health_bar(ax, x, y, health, max_health, color):
    """
    Draws a small health bar above an agent.

    x, y:
        Grid position of the agent.

    health:
        Current health.

    max_health:
        Maximum health.

    color:
        Bar color, for example "green" or "red".
    """
    if max_health <= 0:
        return

    health_ratio = max(0, min(health / max_health, 1))

    bar_width = 0.65
    bar_height = 0.08

    bar_x = x - bar_width / 2
    bar_y = y + 0.38

    # Background bar.
    ax.add_patch(
        plt.Rectangle(
            (bar_x, bar_y),
            bar_width,
            bar_height,
            color="black",
            zorder=10,
        )
    )

    # Current health bar.
    ax.add_patch(
        plt.Rectangle(
            (bar_x, bar_y),
            bar_width * health_ratio,
            bar_height,
            color=color,
            zorder=11,
        )
    )

    # Thin border.
    ax.add_patch(
        plt.Rectangle(
            (bar_x, bar_y),
            bar_width,
            bar_height,
            fill=False,
            edgecolor="white",
            linewidth=0.6,
            zorder=12,
        )
    )

def draw_grid(model):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, top=0.97, bottom=0.01)

    ax.set_xlim(-0.5, model.width - 0.5)
    ax.set_ylim(-0.5, model.height - 0.5)

    background_img = load_image("background")

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
    
    # Pre-compute sizes once per frame
    grid_size = max(model.width, model.height)
    zoom      = image_zoom(grid_size)
    obs_size  = scatter_size(grid_size, base=350)
    safe_size = scatter_size(grid_size, base=500)
    lbl_fs    = font_size(grid_size, base=9.0)

    for agent in list(model.agents):
        if getattr(agent, "pos", None) is None:
            continue

        x, y = agent.pos

        if isinstance(agent, ScoutAgent):
            draw_image(ax, "scout", x, y, zoom=zoom)
            draw_health_bar(ax, x, y, agent.health, agent.max_health, "green")

        elif isinstance(agent, DefenderAgent):
            draw_image(ax, "defender", x, y, zoom=zoom)
            draw_health_bar(ax, x, y, agent.health, agent.max_health, "green")

        elif isinstance(agent, SupportAgent):
            draw_image(ax, "support", x, y, zoom=zoom)
            draw_health_bar(ax, x, y, agent.health, agent.max_health, "green")

        elif isinstance(agent, AdaptiveAgent):
            draw_image(ax, "adaptive", x, y, zoom=zoom)
            draw_health_bar(ax, x, y, agent.health, agent.max_health, "green")

        elif isinstance(agent, ZombieAgent):
            draw_image(ax, "zombie", x, y, zoom=zoom)
            draw_health_bar(ax, x, y, agent.health, agent.max_health, "red")

        elif isinstance(agent, ObstacleAgent):
            ax.scatter(x, y, s=obs_size, marker="s", color="black")

        elif isinstance(agent, SafeZoneAgent):
            ax.scatter(x, y, s=safe_size, marker="P", color="gold")
            ax.text(x, y, "S", ha="center", va="center", color="black", weight="bold", fontsize=lbl_fs)

    return fig


# ==============================
# SURVIVOR PANEL COMPONENT
# ==============================

ROLE_COLORS = {
    "Scout":    {"bg": "#f0f7ee", "accent": "#2d6a4f", "border": "#74c69d", "bar": "#52b788", "icon": "🔭"},
    "Defender": {"bg": "#eef2ff", "accent": "#3730a3", "border": "#818cf8", "bar": "#6366f1", "icon": "🛡️"},
    "Support":  {"bg": "#fff7ed", "accent": "#9a3412", "border": "#fb923c", "bar": "#f97316", "icon": "💊"},
    "Adaptive": {"bg": "#fefce8", "accent": "#854d0e", "border": "#fbbf24", "bar": "#f59e0b", "icon": "⚡"},
}

ACTION_LABELS = {
    "move":   ("🚶", "#3b82f6"),
    "attack": ("⚔️", "#ef4444"),
    "heal":   ("💚", "#22c55e"),
    "scan":   ("📡", "#8b5cf6"),
    "wait":   ("⏸️", "#94a3b8"),
}

def health_color(ratio: float) -> str:
    if ratio > 0.6:
        return "#16a34a"
    elif ratio > 0.3:
        return "#d97706"
    else:
        return "#dc2626"


def agent_card_html(agent) -> str:
    role = getattr(agent, "role_name", "Adaptive")
    c = ROLE_COLORS.get(role, ROLE_COLORS["Adaptive"])
    health = agent.health
    max_health = agent.max_health
    ratio = max(0.0, health / max_health) if max_health > 0 else 0.0
    hp_color = health_color(ratio)
    action_raw = str(getattr(agent, "last_action", "wait") or "wait").lower().replace("action.", "")
    action_icon, action_color = ACTION_LABELS.get(action_raw, ("❓", "#64748b"))
    pos = agent.pos if agent.pos else ("?", "?")
    safe = getattr(agent, "reached_safe_zone", False)
    scanning = getattr(agent, "scanning", False)
    scan_found_safe_zone = getattr(agent, "scan_found_safe_zone", False)

    safe_badge = (
        f"<span style='background:#dcfce7;color:#15803d;font-size:10px;"
        f"padding:2px 8px;border-radius:20px;border:1px solid #86efac;"
        f"font-weight:700;'>✓ Safe Zone</span>"
        if safe else ""
    )

    # segmented HP pips for small max_health (3)
    pips_html = ""
    for i in range(max_health):
        filled = i < health
        pip_color = hp_color if filled else "#e2e8f0"
        pip_border = hp_color if filled else "#cbd5e1"
        pips_html += f"<div style='flex:1;height:12px;background:{pip_color};border:1px solid {pip_border};border-radius:3px;margin:0 2px;'></div>"

    scan_found_safe_zone = getattr(agent, "scan_found_safe_zone", False)
    if scan_found_safe_zone:
        scan_badge = (
            "<span style='background:#ede9fe;color:#7c3aed;padding:2px 7px;"
            "border-radius:6px;font-size:11px;font-weight:600;margin-left:4px;'>📡 Found!</span>"
        )
    elif scanning:
        scan_badge = (
            "<span style='background:#f1f5f9;color:#94a3b8;padding:2px 7px;"
            "border-radius:6px;font-size:11px;font-weight:500;margin-left:4px;'>🔍 Scanning...</span>"
        )
    else:
        scan_badge = ""

    return f"""
<div style="background:{c['bg']};border:1px solid {c['border']};border-radius:10px;
     padding:12px 14px;margin-bottom:8px;font-family:system-ui,sans-serif;
     box-shadow:0 1px 4px rgba(0,0,0,0.08);">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
    <span style="color:{c['accent']};font-weight:700;font-size:14px;">
      {c['icon']} {role}
    </span>
    {safe_badge}
  </div>
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
    <span style="font-size:11px;color:#64748b;min-width:22px;">HP</span>
    <div style="display:flex;flex:1;align-items:center;">
      {pips_html}
    </div>
    <span style="font-size:12px;font-weight:700;color:{hp_color};min-width:28px;text-align:right;">{health}/{max_health}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#64748b;border-top:1px solid {c['border']}40;padding-top:6px;margin-top:2px;flex-wrap:wrap;gap:4px;">
    <span style="background:#f1f5f9;padding:2px 7px;border-radius:6px;font-size:11px;">📍 ({pos[0]}, {pos[1]})</span>
    <div style="display:flex;align-items:center;">
      <span style="background:{action_color}18;color:{action_color};padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600;">{action_icon} {action_raw.upper()}</span>
      {scan_badge}
    </div>
  </div>
</div>
"""


@solara.component
def SurvivorPanel():
    # Subscribe to refresh_counter so this re-renders on every step/play tick
    _ = refresh_counter.value
    model = model_state.value

    survivors = model.get_alive_survivors()
    alive = len(survivors)
    total_hp = sum(a.health for a in survivors)
    max_hp = sum(a.max_health for a in survivors)
    safe_count = sum(1 for a in survivors if getattr(a, "reached_safe_zone", False))
    hp_ratio = total_hp / max_hp if max_hp > 0 else 0
    team_hp_color = health_color(hp_ratio)
    coop = model.cooperation_events

    if not survivors:
        cards_html = "<div style='color:#94a3b8;font-size:13px;text-align:center;padding:24px 0;'>☠️ All survivors eliminated</div>"
    else:
        cards_html = "".join(agent_card_html(a) for a in survivors)

    footer_html = f"""
<div style="border-top:1px solid #e2e8f0;margin-top:4px;padding-top:10px;font-family:system-ui,sans-serif;">
  <div style="font-size:10px;color:#94a3b8;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;">Team Overview</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;">
      <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">Alive</div>
      <div style="font-size:18px;font-weight:700;color:#1e293b;">{alive}</div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;">
      <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">Team HP</div>
      <div style="font-size:18px;font-weight:700;color:{team_hp_color};">{total_hp}<span style="font-size:11px;color:#94a3b8;">/{max_hp}</span></div>
    </div>
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:8px 10px;">
      <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">In Safe Zone</div>
      <div style="font-size:18px;font-weight:700;color:#16a34a;">{safe_count}</div>
    </div>
    <div style="background:#fef9c3;border:1px solid #fde68a;border-radius:8px;padding:8px 10px;">
      <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">Cooperations</div>
      <div style="font-size:18px;font-weight:700;color:#d97706;">{coop}</div>
    </div>
  </div>
</div>
"""

    full_html = f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;
     padding:18px;width:280px;font-family:system-ui,sans-serif;
     box-shadow:0 2px 12px rgba(0,0,0,0.07);">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid #e2e8f0;">
    <span style="font-size:18px;">👥</span>
    <span style="font-weight:700;font-size:14px;color:#1e293b;letter-spacing:0.3px;">Survivor Agents</span>
  </div>
  {cards_html}
  {footer_html}
</div>
"""

    solara.HTML(tag="div", unsafe_innerHTML=full_html)


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
                "adaptive_adaptive_adaptive",
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

    with solara.Row(style={"alignItems": "flex-start", "gap": "24px"}):
        with solara.Column():
            fig = draw_grid(model)
            solara.FigureMatplotlib(fig)
            plt.close(fig)

        SurvivorPanel()