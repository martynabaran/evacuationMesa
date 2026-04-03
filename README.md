# Evacuation Simulation — Mesa

An agent-based simulation of building evacuation under fire conditions, built with the [Mesa](https://mesa.readthedocs.io/) framework.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Experiments](#running-experiments)
- [Output Files](#output-files)
- [Agents](#agents)
- [Environment](#environment)
- [Metrics](#metrics)

---

## Overview

The simulation models a group of human agents evacuating a building during a fire. Key features:

- **Fire and smoke spread** via a cellular automaton (probabilistic, numpy-based)
- **Three evacuation strategies** per agent: `nearest_exit`, `safest_exit`, `least_crowded_exit`
- **Age-based movement speed** — older agents move slower
- **Risk tolerance** — determines willingness to cross smoke/fire
- **Family groups** — agents may pause to rescue relatives before evacuating
- **Communication** — some agents broadcast hazard locations to nearby agents
- **Alarm system** — agents switch from idle to evacuation mode when triggered
- **A\* pathfinding** with per-agent cost weights; shared precomputed exit distance maps for efficiency

---

## Project Structure

```
evacuation_mesa/
├── run.py            # Entry point — CLI for all experiment modes
├── model.py          # EvacuationModel: top-level Mesa Model
├── agents.py         # EvacuationAgent: agent state machine and behaviours
├── environment.py    # Grid, fire/smoke CA, exit distance maps
├── pathfinding.py    # A* and random walk implementations
├── metrics.py        # Mesa DataCollector definitions
├── config.py         # Configuration dataclasses and YAML loader
├── requirements.txt  # Python dependencies
└── room_layouts/     # Text files defining building layouts (expected by config)
```

---

## Setup

**1. Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Configuration

Simulation parameters are loaded from a YAML file (default: `config.yaml`). All keys are optional — missing values fall back to built-in defaults.

Example `config.yaml`:

```yaml
simulation:
  num_agents: 30
  seed: 42
  max_steps: 2000
  num_families: 8
  rescue_timeout: 30

  agent_attributes:
    health: 100
    risk: [0.1, 0.9]          # uniform range for risk tolerance per agent
    range: 5                   # perception range (cells)
    volume: 5                  # communication range (cells)
    communicates: 30           # number of communicative agents
    layout: room_layouts/supermarket3.txt

  strategies:
    nearest_exit: 10
    safest_exit: 10
    least_crowded_exit: 10

  fire:
    fire_probability: 0.6      # per-step spread probability
    smoke_probability: 0.3
    fire_damage: 50            # health points lost per step in fire
    smoke_damage: 2            # health points lost per step in smoke
```

### Layout files

Building layouts are plain text grids. Each cell is one of:

| Character | Meaning |
|-----------|---------|
| `O`       | Open passable floor |
| `W`       | Wall (impassable) |
| `E`       | Exit |
| `F`       | Initial fire cell |
| `S`       | Initial smoke cell |
| `A`       | Alarm sensor |

---

## Running Experiments

### Single headless run

Runs the simulation without any display. Results are saved to `results/`.

```bash
python run.py
```

Use a custom config file:

```bash
python run.py --config my_config.yaml
```

### Single run with live visualization

Renders the grid and live metrics charts using matplotlib.

```bash
python run.py --visualize
```

### Batch run (parameter sweep)

Runs the model N times with different random seeds, collecting final outcomes. Results are saved to `results/batch_results.csv`.

```bash
python run.py --batch                  # default: 50 replications
python run.py --batch --n-reps 100     # custom number of replications
```

---

## Output Files

After a single run, two CSV files are written to `results/`:

| File | Contents |
|------|----------|
| `model_timeseries.csv` | One row per step: escaped, dead, alive, fire cells, smoke cells, mean health, evacuation rate, panicking agents, rescuing agents |
| `agent_data.csv` | One row per agent per step: state, health, position, age, risk, strategy, family ID, escape step, cells traversed, hazard cells crossed, steps in smoke/fire, communications received |

After a batch run:

| File | Contents |
|------|----------|
| `results/batch_results.csv` | One row per replication with final model-level metrics |

---

## Agents

Each agent has the following attributes assigned at initialization:

| Attribute | Description |
|-----------|-------------|
| `age` | Sampled uniformly from `[age_min, age_max]`; determines movement speed |
| `risk` | Sampled uniformly from `risk_range`; higher = more willing to cross smoke/fire |
| `strategy` | One of `nearest_exit`, `safest_exit`, `least_crowded_exit` |
| `communicates` | Whether the agent broadcasts known hazard positions to nearby agents |
| `family_id` | Group ID if part of a family; family members always communicate |

### Agent state machine

```
idle → aware → evacuating → escaped
                   ↓
               rescuing → aware
                   ↓
               panic → aware (retried every 5 steps)
                   ↓
               dead
```

- **idle** — pre-alarm; performs random walk
- **aware** — plans evacuation path using chosen strategy
- **evacuating** — follows A\* path to chosen exit
- **rescuing** — moves toward an idle relative before evacuating
- **panic** — no path found; random walk, periodically retries pathfinding
- **escaped** — reached an exit cell
- **dead** — health reached zero

---

## Environment

The grid is a shared numpy array of integer cell types. Fire and smoke spread via a vectorized cellular automaton:

- **Fire** spreads to adjacent open/smoke cells with probability `fire_spread_prob` every `fire_interval` steps
- **Smoke** spreads from fire and smoke cells to adjacent open cells with probability `smoke_spread_prob` every `smoke_interval` steps

Exit distance maps (one per exit) are precomputed via reverse Dijkstra and shared across all agents. They are recomputed lazily only when fire/smoke configuration changes.

---

## Metrics

Model-level metrics collected every step:

| Metric | Description |
|--------|-------------|
| `Escaped` | Cumulative escaped agents |
| `Dead` | Cumulative dead agents |
| `Alive` | Agents still inside |
| `FireCells` | Number of fire cells on the grid |
| `SmokeCells` | Number of smoke cells on the grid |
| `MeanHealth` | Average health of active agents |
| `EvacuationRate` | `Escaped / total_agents` |
| `Panicking` | Agents in panic state |
| `Rescuing` | Agents currently rescuing a relative |
