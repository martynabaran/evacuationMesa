"""
DataCollector definitions for the Mesa evacuation simulation.

Collected at two levels:
  - model-level:  aggregates across all agents per step
  - agent-level:  per-agent row collected at the end of simulation

Why this is better than the original CSV approach:
  - Collected through Mesa's DataCollector — works with BatchRunner automatically.
  - Agent-level data includes path quality metrics (hazard cells crossed,
    cells traversed, steps in smoke/fire) that the original did not capture.
  - Model-level time series allows evacuation curve analysis (how many agents
    escaped by step N).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from environment import FIRE, SMOKE

if TYPE_CHECKING:
    from model import EvacuationModel


# ------------------------------------------------------------------ #
#  Model-level reporters (called every step)                          #
# ------------------------------------------------------------------ #

def count_escaped(model: "EvacuationModel") -> int:
    return len(model.escaped_agents)


def count_dead(model: "EvacuationModel") -> int:
    return len(model.dead_agents)


def count_alive(model: "EvacuationModel") -> int:
    return sum(
        1 for a in model.schedule.agents
        if a.state not in ("escaped", "dead")
    )


def count_fire_cells(model: "EvacuationModel") -> int:
    return int(np.sum(model.env.grid == FIRE))


def count_smoke_cells(model: "EvacuationModel") -> int:
    return int(np.sum(model.env.grid == SMOKE))


def mean_health(model: "EvacuationModel") -> float:
    healths = [a.health for a in model.schedule.agents if a.state not in ("escaped", "dead")]
    return float(np.mean(healths)) if healths else 0.0


def evacuation_rate(model: "EvacuationModel") -> float:
    total = model.cfg.num_agents
    return len(model.escaped_agents) / total if total > 0 else 0.0


def agents_in_panic(model: "EvacuationModel") -> int:
    return sum(1 for a in model.schedule.agents if a.state == "panic")


def agents_rescuing(model: "EvacuationModel") -> int:
    return sum(1 for a in model.schedule.agents if a.state == "rescuing")


# ------------------------------------------------------------------ #
#  Agent-level reporters (collected at end of run or per step)        #
# ------------------------------------------------------------------ #

def agent_state(agent) -> str:
    return agent.state


def agent_health(agent) -> int:
    return agent.health


def agent_pos_r(agent) -> int:
    return agent.pos[0]


def agent_pos_c(agent) -> int:
    return agent.pos[1]


# ------------------------------------------------------------------ #
#  DataCollector factory                                              #
# ------------------------------------------------------------------ #

def make_data_collector():
    """
    Returns a configured mesa.DataCollector instance.
    Import here to avoid circular import at module level.
    """
    import mesa

    return mesa.DataCollector(
        model_reporters={
            "Escaped":        count_escaped,
            "Dead":           count_dead,
            "Alive":          count_alive,
            "FireCells":      count_fire_cells,
            "SmokeCells":     count_smoke_cells,
            "MeanHealth":     mean_health,
            "EvacuationRate": evacuation_rate,
            "Panicking":      agents_in_panic,
            "Rescuing":       agents_rescuing,
        },
        agent_reporters={
            "State":               agent_state,
            "Health":              agent_health,
            "Row":                 agent_pos_r,
            "Col":                 agent_pos_c,
            "Age":                 "age",
            "Risk":                "risk",
            "Strategy":            "strategy",
            "FamilyID":            "family_id",
            "EscapeStep":          "escape_step",
            "CellsTraversed":      "cells_traversed",
            "HazardCellsCrossed":  "hazard_cells_crossed",
            "StepsInSmoke":        "steps_in_smoke",
            "StepsInFire":         "steps_in_fire",
            "CommsReceived":       "communications_received",
            "RescueCompleted":     "rescue_completed",
            "ExitChosen_r":        lambda a: a.exit_chosen[0] if a.exit_chosen else -1,
            "ExitChosen_c":        lambda a: a.exit_chosen[1] if a.exit_chosen else -1,
        },
    )
