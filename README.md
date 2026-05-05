# AASMA-Project-25-26

## Ad Hoc Teamwork in a Zombie Survival Grid World

This project explores ad hoc teamwork in a multi-agent zombie survival grid world.

The goal is to study how an adaptive autonomous agent can cooperate with unknown teammates without previous coordination. The environment is a grid world with survivor agents, zombies, obstacles, and a safe zone.

Project for Autonomous Agents and Multi-Agent Systems in IST by Francisco Nascimento, Miguel Baptista and Daniel Barbosa.

## Project Idea

A team of survivor agents must survive, avoid or fight zombies, and reach a safe zone.

The survivor team can include different agent profiles:

- **Scout**: focuses on moving safely toward the safe zone.
- **Defender**: attacks or blocks zombies to protect the team.
- **Support**: stays close to teammates and heals injured agents.
- **Adaptive Agent**: observes unknown teammates and adapts its behaviour to complement the team.

The adaptive agent does not know the roles of its teammates in advance. It observes their actions and estimates whether they behave more like a Scout, Defender, or Support.

## Project Structure

```text
AASMA-Project-25-26/
│
├── main.py                         # Runs all experiments without visualization
├── app.py                          # Mesa/Solara visualization
├── requirements.txt                # Python dependencies
├── README.md                       # Instructions
│
├── config/
│   └── config.py                   # Global configuration parameters
│
├── src/
│   ├── model.py                    # Defines the Mesa model and grid environment
│   │
│   ├── agents/
│   │   ├── survivor_agents.py      # Scout, Defender, Support and Adaptive agents
│   │   ├── zombie_agent.py         # Zombie agent behaviour
│   │   └── environment_agents.py   # Obstacles and safe zone
│   │
│   ├── experiments/
│   │   └── experiments.py          # Defines and runs experiment configurations
│   │
│   ├── metrics/
│   │   └── metrics.py              # Metrics and result aggregation
│   │
│   └── utils/
│       └── utils.py                # Utility functions, such as grid distance and movement
│
└── debug/
    └── debug_model.py

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
cd src
```

```bash
python main.py
```

This compares the following team configurations:

```text
Baseline: Scout + Defender + Support
Adaptive replaces Scout
Adaptive replaces Defender
Adaptive replaces Support
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
