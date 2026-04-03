"""
Entry point for the Mesa evacuation simulation.

Modes:
  1. Single headless run           python run.py
  2. Single run with live pyplot   python run.py --visualize
  3. Parameter sweep / batch run   python run.py --batch

The separation of Model from visualization means tests and batch runs
never require a display — a key improvement over the original Pygame design.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import SimulationConfig, load_config
from model import EvacuationModel


# ------------------------------------------------------------------ #
#  Single run                                                         #
# ------------------------------------------------------------------ #

def run_single(cfg: SimulationConfig, visualize: bool = False) -> EvacuationModel:
    model = EvacuationModel(cfg)

    if visualize:
        _run_with_visualization(model)
    else:
        _run_headless(model)

    model.export_results("results")
    _print_summary(model)
    return model


def _run_headless(model: EvacuationModel) -> None:
    t0 = time.perf_counter()
    while model.running:
        model.step()
    elapsed = time.perf_counter() - t0
    print(f"Simulation finished in {model.schedule.steps} steps ({elapsed:.2f}s real time)")


def _run_with_visualization(model: EvacuationModel) -> None:
    """
    Lightweight matplotlib visualization — no Pygame dependency.
    Renders grid state every N steps.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    from environment import OPEN, WALL, EXIT, FIRE, SMOKE, ALARM

    CMAP_VALS = {
        OPEN:  (1.0, 1.0, 1.0),   # white
        WALL:  (0.1, 0.3, 0.6),   # blue
        EXIT:  (0.0, 0.8, 0.0),   # green
        FIRE:  (1.0, 0.4, 0.0),   # orange
        SMOKE: (0.5, 0.5, 0.5),   # grey
        ALARM: (0.0, 0.9, 0.9),   # cyan
    }

    RENDER_EVERY = 3  # steps between renders

    plt.ion()
    fig, (ax_grid, ax_stats) = plt.subplots(1, 2, figsize=(14, 6))
    ax_grid.set_title("Evacuation Simulation")
    ax_stats.set_title("Live Metrics")

    while model.running:
        model.step()

        if model.schedule.steps % RENDER_EVERY == 0:
            # Build RGB image from grid
            grid = model.env.grid
            rows, cols = grid.shape
            img = np.ones((rows, cols, 3))
            for cell_type, color in CMAP_VALS.items():
                mask = grid == cell_type
                img[mask] = color

            # Draw agents on top
            for agent in model.schedule.agents:
                if agent.state in ("escaped", "dead"):
                    continue
                r, c = agent.pos
                strategy_colors = {
                    "nearest_exit":       (0.8, 0.1, 0.1),
                    "safest_exit":        (0.5, 0.0, 0.5),
                    "least_crowded_exit": (0.1, 0.5, 0.8),
                }
                img[r, c] = strategy_colors.get(agent.strategy, (0.2, 0.2, 0.2))

            ax_grid.clear()
            ax_grid.imshow(img, origin="upper", interpolation="nearest")
            ax_grid.set_title(
                f"Step {model.schedule.steps} | "
                f"Escaped: {len(model.escaped_agents)} | "
                f"Dead: {len(model.dead_agents)}"
            )

            # Stats plot
            mdf = model.datacollector.get_model_vars_dataframe()
            ax_stats.clear()
            ax_stats.plot(mdf["Escaped"],  label="Escaped",  color="green")
            ax_stats.plot(mdf["Dead"],     label="Dead",     color="red")
            ax_stats.plot(mdf["Alive"],    label="Alive",    color="blue")
            ax_stats.plot(mdf["FireCells"],  label="Fire cells",  color="orange", linestyle="--")
            ax_stats.plot(mdf["SmokeCells"], label="Smoke cells", color="grey",   linestyle="--")
            ax_stats.legend(fontsize=8)
            ax_stats.set_xlabel("Step")

            plt.tight_layout()
            plt.pause(0.05)

    plt.ioff()
    plt.show()


def _print_summary(model: EvacuationModel) -> None:
    total = model.cfg.num_agents
    esc   = len(model.escaped_agents)
    dead  = len(model.dead_agents)
    alive = total - esc - dead

    print("\n=== Simulation Summary ===")
    print(f"  Total agents : {total}")
    print(f"  Escaped      : {esc}  ({100*esc/total:.1f}%)")
    print(f"  Dead         : {dead} ({100*dead/total:.1f}%)")
    print(f"  Still inside : {alive}")

    if model.escaped_agents:
        esc_times = [a.escape_step for a in model.escaped_agents]
        print(f"  Avg escape step: {np.mean(esc_times):.1f}  (min={min(esc_times)}, max={max(esc_times)})")

    by_strategy: Dict[str, Dict[str, int]] = {}
    for a in model.escaped_agents + model.dead_agents:
        s = a.strategy
        if s not in by_strategy:
            by_strategy[s] = {"escaped": 0, "dead": 0}
        key = "escaped" if a in model.escaped_agents else "dead"
        by_strategy[s][key] += 1

    print("\n  Strategy breakdown:")
    for strat, counts in by_strategy.items():
        print(f"    {strat:25s}  escaped={counts['escaped']}  dead={counts['dead']}")


# ------------------------------------------------------------------ #
#  Batch runner (parameter sweep)                                     #
# ------------------------------------------------------------------ #

def run_batch(n_replications: int = 50) -> pd.DataFrame:
    """
    Run the model N times with fixed config but different seeds,
    collecting final outcome per run.

    For full parameter sweeps use mesa.batch_run (Mesa ≥ 1.0):
      from mesa.batchrunner import batch_run
      results = batch_run(EvacuationModel, parameters={...}, iterations=50)
    """
    from mesa.batchrunner import batch_run

    cfg = load_config()

    parameters = {
        # Vary seed for stochastic replication
        "cfg": [
            SimulationConfig(
                layout_path=cfg.layout_path,
                num_agents=cfg.num_agents,
                seed=seed,
                agent=cfg.agent,
                family=cfg.family,
                fire=cfg.fire,
            )
            for seed in range(n_replications)
        ]
    }

    results = batch_run(
        EvacuationModel,
        parameters=parameters,
        iterations=1,
        max_steps=cfg.max_steps,
        data_collection_period=-1,   # collect only at end
        display_progress=True,
    )

    df = pd.DataFrame(results)
    df.to_csv("results/batch_results.csv", index=False)
    print(f"Batch run complete: {n_replications} replications written to results/batch_results.csv")
    return df


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Mesa Evacuation Simulation")
    parser.add_argument("--config",     default="config.yaml",  help="Path to config YAML")
    parser.add_argument("--visualize",  action="store_true",     help="Show matplotlib visualization")
    parser.add_argument("--batch",      action="store_true",     help="Run batch parameter sweep")
    parser.add_argument("--n-reps",     type=int, default=50,    help="Number of batch replications")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.batch:
        run_batch(n_replications=args.n_reps)
    else:
        run_single(cfg, visualize=args.visualize)


if __name__ == "__main__":
    main()
