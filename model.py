"""
EvacuationModel — the top-level Mesa Model.

Responsibilities:
  - Initialize grid, fire environment, exit distance maps
  - Place agents with correct attribute distributions
  - Form family groups
  - Run fire/smoke CA at configured intervals
  - Check alarm activation
  - Run Mesa DataCollector
  - Expose termination condition

The model is fully headless — visualization is optional (see run.py).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Set, Tuple

import mesa
import numpy as np

from config import SimulationConfig, load_config
from environment import (
    FireEnvironment, ExitDistanceMaps,
    load_layout, get_exits, get_alarms,
    OPEN, WALL, EXIT, FIRE, SMOKE, ALARM,
)
from agents import EvacuationAgent
from metrics import make_data_collector


class EvacuationModel(mesa.Model):
    """
    Parameters
    ----------
    cfg : SimulationConfig
        Full configuration dataclass. Pass explicitly or use load_config().
    """

    def __init__(self, cfg: Optional[SimulationConfig] = None):
        super().__init__()
        self.cfg = cfg or load_config()

        # Seeded RNG for reproducibility
        self.random = random.Random(self.cfg.seed)
        self._np_rng = np.random.default_rng(self.cfg.seed)

        # --- Load environment ---
        layout_path = self.cfg.layout_path
        if not os.path.exists(layout_path):
            # Try relative to this file
            here = os.path.dirname(__file__)
            layout_path = os.path.join(here, "..", "SSD_2024", layout_path)
        grid = load_layout(layout_path)

        self.env = FireEnvironment(grid, self._np_rng)
        self.exit_positions: List[Tuple[int, int]] = get_exits(grid)
        self.alarm_positions: List[Tuple[int, int]] = get_alarms(grid)

        # Precompute exit distance maps — O(exits × V log V) once at init
        self.exit_maps = ExitDistanceMaps(grid, self.exit_positions)

        # --- Scheduler ---
        # RandomActivation: each step, all agents act in a random order
        # Use SimultaneousActivation for synchronous updates (more realistic
        # for crowd dynamics but slightly slower)
        self.schedule = mesa.time.RandomActivation(self)

        # --- Cell occupancy map for O(1) collision checks ---
        self.cell_occupant: Dict[Tuple[int, int], EvacuationAgent] = {}

        # --- Alarm state ---
        self.alarm_active: bool = False

        # --- Tracking ---
        self.escaped_agents: List[EvacuationAgent] = []
        self.dead_agents: List[EvacuationAgent] = []

        # Quick lookup by id
        self.agent_by_id: Dict[int, EvacuationAgent] = {}

        # --- Place agents ---
        self._initialize_agents()
        self._initialize_families()

        # --- Start fire ---
        self._start_fire()

        # --- DataCollector ---
        self.datacollector = make_data_collector()
        self.datacollector.collect(self)

        self.running = True

    # ------------------------------------------------------------------ #
    #  Initialization helpers                                             #
    # ------------------------------------------------------------------ #

    def _passable_cells(self) -> List[Tuple[int, int]]:
        """All cells where an agent can be placed at init."""
        rows, cols = self.env.grid.shape
        cells = []
        for r in range(rows):
            for c in range(cols):
                if self.env.grid[r, c] == OPEN:
                    cells.append((r, c))
        return cells

    def _initialize_agents(self) -> None:
        cfg = self.cfg
        acfg = cfg.agent

        passable = self._passable_cells()
        self.random.shuffle(passable)

        # Build strategy pool in order
        strategy_pool: List[str] = []
        for strategy, count in acfg.strategies.items():
            strategy_pool.extend([strategy] * count)
        # Pad or trim to num_agents
        while len(strategy_pool) < cfg.num_agents:
            strategy_pool.append(self.random.choice(list(acfg.strategies.keys())))
        strategy_pool = strategy_pool[:cfg.num_agents]
        self.random.shuffle(strategy_pool)

        for i in range(cfg.num_agents):
            pos = passable[i % len(passable)]
            risk = self.random.uniform(*acfg.risk_range)
            age  = self.random.randint(cfg.age_min, cfg.age_max)
            communicates = (i < cfg.num_communicative)

            agent = EvacuationAgent(
                unique_id=i,
                model=self,
                pos=pos,
                health=acfg.health,
                risk=risk,
                age=age,
                communicates=communicates,
                strategy=strategy_pool[i],
                perception_range=acfg.perception_range,
                communication_range=acfg.communication_range,
                rescue_timeout=cfg.family.rescue_timeout,
            )
            self.schedule.add(agent)
            self.cell_occupant[pos] = agent
            self.agent_by_id[i] = agent

    def _initialize_families(self) -> None:
        fcfg = self.cfg.family
        all_agents = list(self.schedule.agents)
        self.random.shuffle(all_agents)

        remaining = list(all_agents)
        for family_id in range(fcfg.num_families):
            if len(remaining) < fcfg.min_size:
                break
            size = self.random.randint(fcfg.min_size, fcfg.max_size)
            size = min(size, len(remaining))
            members = [remaining.pop() for _ in range(size)]

            for agent in members:
                agent.family_id = family_id
                agent.communicates = True  # family members always communicate
                for other in members:
                    if other.unique_id != agent.unique_id:
                        agent.relatives.append(other.unique_id)

    def _start_fire(self) -> None:
        passable = self._passable_cells()
        r, c = self.random.choice(passable)
        self.env.ignite(r, c)

    # ------------------------------------------------------------------ #
    #  Mesa step                                                          #
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        step = self.schedule.steps

        # 1. Fire/smoke propagation (at configured intervals)
        if step % self.cfg.fire.fire_interval == 0:
            new_fire = self.env.propagate_fire(self.cfg.fire.fire_spread_prob)
            if new_fire:
                # Invalidate and recompute affected exit distance maps
                self.exit_maps.maybe_recompute(self.env.grid)

        if step % self.cfg.fire.smoke_interval == 0:
            new_smoke = self.env.propagate_smoke(self.cfg.fire.smoke_spread_prob)
            if new_smoke:
                self.exit_maps.maybe_recompute(self.env.grid)

        # 2. Alarm check
        if not self.alarm_active:
            if self.env.check_alarms(self.alarm_positions):
                self.alarm_active = True

        # 3. Communication broadcast (communicative agents only)
        for agent in list(self.schedule.agents):
            if agent.state not in ("escaped", "dead") and agent.communicates:
                agent.broadcast_hazards()

        # 4. Agent steps
        self.schedule.step()

        # 5. Collect metrics
        self.datacollector.collect(self)

        # 6. Termination check
        alive = sum(1 for a in self.schedule.agents if a.state not in ("escaped", "dead"))
        if alive == 0 or step >= self.cfg.max_steps:
            self.running = False

    # ------------------------------------------------------------------ #
    #  Event registration                                                 #
    # ------------------------------------------------------------------ #

    def register_escaped(self, agent: EvacuationAgent) -> None:
        self.escaped_agents.append(agent)

    def register_dead(self, agent: EvacuationAgent) -> None:
        self.dead_agents.append(agent)

    # ------------------------------------------------------------------ #
    #  Results export                                                     #
    # ------------------------------------------------------------------ #

    def export_results(self, output_dir: str = ".") -> None:
        """Write model and agent DataFrames to CSV."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        model_df = self.datacollector.get_model_vars_dataframe()
        agent_df = self.datacollector.get_agent_vars_dataframe()

        model_df.to_csv(os.path.join(output_dir, "model_timeseries.csv"))
        agent_df.to_csv(os.path.join(output_dir, "agent_data.csv"))
        print(f"Results written to {output_dir}/")
