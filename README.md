# AASMA-Project-25-26

## Ad Hoc Teamwork in a Zombie Survival Grid World

This repository studies ad hoc teamwork in a multi-agent zombie survival grid world.
An adaptive agent must cooperate with unknown teammates (Scout, Defender, Support) without prior coordination. The environment contains survivor agents, zombies, obstacles, and a safe zone.

Project authors: Francisco Nascimento, Miguel Baptista and Daniel Barbosa.

## Project Structure

```text
AASMA-Project-25-26/
├── main.py                 # Run experiments (no visualization)
├── compare_rl.py           # Scripts to compare RL policies
├── app.py                  # Mesa/Solara visualization
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── config/
│   └── config.py           # Global configuration parameters
├── images/                 # Images used by the app
├── results/                # Experiment outputs and logs
└── src/
		├── model.py            # Mesa model and grid environment
		├── experiments.py      # Experiment setups and runners
		├── metrics.py          # Metrics aggregation and helpers
        ├── analysis.py         # Data analysis to build graphs and charts
		├── utils.py            # Utility helpers (distance, movement, etc.)
		└── agents/
				├── survivor_agents.py    # Scout, Defender, Support behaviours
				├── adaptive_agent.py     # Adaptive agent implementation
				├── rl_policy.py          # RL policy / helper code
				├── zombie_agent.py       # Zombie behaviour
				└── environment_agents.py # Obstacles, safe zone, and environment actors
```

## Installation and Running the Project

### 1. Open the terminal in the project folder

```bash
cd AASMA-Project-25-26
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```bash
venv\Scripts\Activate.ps1
```

After activation, the terminal should show something like:

```text
(venv) ...
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Experiments

To run the experiments without visualization:

```bash
python main.py
```

This compares the following team configurations:

```text
Baseline: Scout + Defender + Support
Adaptive replaces Scout
Adaptive replaces Defender
Adaptive replaces Support
Team: Adaptive + Adaptive + Adaptive
```

## Running the Visualization

To run the interactive Mesa/Solara visualization:

```bash
solara run app.py
```

Then open the local URL shown in the terminal.

Usually it will be something like:

```text
http://localhost:8765
```

## Deactivating the Virtual Environment

When you are done, deactivate the virtual environment with:

```bash
deactivate
```