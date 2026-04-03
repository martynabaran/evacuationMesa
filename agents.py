"""
EvacuationAgent — Mesa Agent for the fire evacuation simulation.

Design improvements over the original:

1. No layout copy per agent — agents read from model.env.grid (shared numpy array).
2. Age-based speed implemented properly via a fractional step accumulator.
3. Risk tolerance affects both smoke AND fire routing.
4. Family rescue has a configurable timeout (rescue_timeout steps).
5. Communicates only hazard deltas (positions), not entire grid copies.
6. Rich per-agent metrics stored for DataCollector.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import mesa

from environment import (
    OPEN, WALL, EXIT, FIRE, SMOKE, ALARM, NEIGHBORS
)
from pathfinding import astar, random_walk


class EvacuationAgent(mesa.Agent):
    """
    Human agent that evacuates a building during a fire.

    States
    ------
    idle        : pre-alarm, random walk
    aware       : alarm triggered, computing evacuation path
    evacuating  : following path to chosen exit
    rescuing    : moving toward a relative before evacuating
    panic       : no path found, random walk allowing smoke
    escaped     : reached exit — removed from scheduler next step
    dead        : health <= 0 — removed from scheduler next step
    """

    def __init__(
        self,
        unique_id: int,
        model: "EvacuationModel",
        pos: Tuple[int, int],
        health: int,
        risk: float,
        age: int,
        communicates: bool,
        strategy: str,
        perception_range: int,
        communication_range: int,
        rescue_timeout: int,
    ):
        super().__init__(unique_id, model)
        self.pos: Tuple[int, int] = pos
        self.health: int = health
        self.risk: float = risk
        self.age: int = age
        self.communicates: bool = communicates
        self.strategy: str = strategy
        self.perception_range: int = perception_range
        self.communication_range: int = communication_range
        self.rescue_timeout: int = rescue_timeout

        # Speed: linear interpolation over age range, stored as fraction per step
        # e.g. 1.0 means 1 cell/step, 0.5 means 1 cell every 2 steps
        cfg = model.cfg
        t = (age - cfg.age_min) / max(cfg.age_max - cfg.age_min, 1)
        self.speed: float = cfg.speed_young + t * (cfg.speed_old - cfg.speed_young)
        self._speed_acc: float = 0.0  # accumulator for fractional movement

        # State machine
        self.state: str = "idle"
        self._path: List[Tuple[int, int]] = []
        self._target_exit: Optional[Tuple[int, int]] = None

        # Awareness: set of (r, c) of known hazard cells
        self._known_hazards: Set[Tuple[int, int]] = set()

        # Family
        self.family_id: Optional[int] = None
        self.relatives: List[int] = []          # unique_ids of family members
        self._rescue_target_id: Optional[int] = None
        self._rescue_steps: int = 0

        # Metrics (populated as simulation runs)
        self.escape_step: int = -1
        self.steps_in_smoke: int = 0
        self.steps_in_fire: int = 0
        self.cells_traversed: int = 0
        self.hazard_cells_crossed: int = 0
        self.communications_received: int = 0
        self.rescue_completed: bool = False
        self.exit_chosen: Optional[Tuple[int, int]] = None
        self._last_pos: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    #  Mesa step method — called by scheduler each tick                   #
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        if self.state in ("escaped", "dead"):
            return

        grid = self.model.env.grid
        cur_cell = grid[self.pos[0], self.pos[1]]

        # --- Health update ---
        self._update_health(cur_cell)
        if self.state == "dead":
            return

        # --- Perception: scan neighbourhood, update known hazards ---
        self._perceive()

        # --- State transitions ---
        if self.state == "idle":
            if self.model.alarm_active or self._known_hazards:
                self.state = "aware"
            else:
                self._random_move()
                return

        if self.state == "aware":
            self._plan_evacuation()

        if self.state == "rescuing":
            self._do_rescue()
            return  # rescue logic handles its own movement

        if self.state == "evacuating":
            self._follow_path()

        if self.state == "panic":
            self._panic_move()

    # ------------------------------------------------------------------ #
    #  Perception                                                          #
    # ------------------------------------------------------------------ #

    def _perceive(self) -> None:
        grid = self.model.env.grid
        rows, cols = grid.shape
        r, c = self.pos
        pr = self.perception_range

        for dr in range(-pr, pr + 1):
            for dc in range(-pr, pr + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    cell = grid[nr, nc]
                    if cell in (FIRE, SMOKE):
                        self._known_hazards.add((nr, nc))
                        if self.state == "evacuating":
                            # Invalidate path if hazard appeared on it
                            if (nr, nc) in self._path:
                                self._path = []
                                self.state = "aware"

    # ------------------------------------------------------------------ #
    #  Communication                                                       #
    # ------------------------------------------------------------------ #

    def receive_hazards(self, hazard_positions: Set[Tuple[int, int]]) -> None:
        """Accept hazard positions broadcast by a nearby communicative agent."""
        new = hazard_positions - self._known_hazards
        if new:
            self._known_hazards |= new
            self.communications_received += 1
            if self.state == "idle":
                self.state = "aware"
            elif self.state == "evacuating" and any(h in self._path for h in new):
                self._path = []
                self.state = "aware"

    def broadcast_hazards(self) -> None:
        """Broadcast known hazards to agents within communication_range."""
        if not self.communicates or not self._known_hazards:
            return
        r, c = self.pos
        cr = self.communication_range
        for agent in self.model.schedule.agents:
            if agent.unique_id == self.unique_id:
                continue
            if agent.state in ("escaped", "dead"):
                continue
            ar, ac = agent.pos
            if abs(ar - r) <= cr and abs(ac - c) <= cr:
                agent.receive_hazards(self._known_hazards)

    # ------------------------------------------------------------------ #
    #  Planning                                                            #
    # ------------------------------------------------------------------ #

    def _plan_evacuation(self) -> None:
        """
        Choose exit and compute path.
        Uses shared DistanceMaps for exit selection (O(1) per exit lookup),
        then A* only if the shared map next-step is unavailable or overridden.
        """
        all_positions = {a.pos for a in self.model.schedule.agents
                         if a.state not in ("escaped", "dead")}

        target = self.model.exit_maps.best_exit_for(
            self.pos,
            strategy=self.strategy,
            agent_risk=self.risk,
            agent_grid=self.model.env.grid,
            all_agent_positions=all_positions,
        )
        self._target_exit = target
        self.exit_chosen = target

        # Check if a relative needs rescuing first
        if self.relatives and self.state != "rescuing":
            rescue_target = self._find_relative_in_danger()
            if rescue_target is not None:
                self._rescue_target_id = rescue_target
                self._rescue_steps = 0
                self.state = "rescuing"
                return

        # Compute path via A*
        path = astar(
            self.model.env.grid,
            self.pos,
            target,
            risk_tolerance=self.risk,
            allow_fire=False,
        )
        if path:
            self._path = path[1:]  # exclude current cell
            self.state = "evacuating"
        else:
            # No path — try allowing fire (panic)
            path = astar(
                self.model.env.grid,
                self.pos,
                target,
                risk_tolerance=self.risk,
                allow_fire=True,
            )
            if path:
                self._path = path[1:]
                self.state = "evacuating"
            else:
                self.state = "panic"

    # ------------------------------------------------------------------ #
    #  Movement                                                            #
    # ------------------------------------------------------------------ #

    def _should_move_this_step(self) -> bool:
        """
        Returns True if the agent should move this step based on speed.
        speed=1.0 → moves every step
        speed=0.5 → moves every 2 steps
        """
        self._speed_acc += self.speed
        if self._speed_acc >= 1.0:
            self._speed_acc -= 1.0
            return True
        return False

    def _follow_path(self) -> None:
        if not self._path:
            self.state = "aware"
            return

        if not self._should_move_this_step():
            return

        next_cell = self._path[0]

        # Collision resolution: if occupied, try to yield or skip one step
        if not self._try_move(next_cell):
            # Try adjacent free cell toward exit (partial reroute)
            alt = self._find_alternative_step(next_cell)
            if alt:
                self._try_move(alt)
            # else stay put this step — no deadlock, just delay

    def _try_move(self, target: Tuple[int, int]) -> bool:
        """
        Attempt to move to target.
        Returns True on success, False if blocked.
        Two agents moving toward each other are allowed to swap (avoids deadlock).
        """
        occupant = self.model.cell_occupant.get(target)
        if occupant is not None and occupant is not self:
            # Allow swap: if occupant's next step is this agent's current pos
            occ_next = occupant._path[0] if occupant._path else None
            if occ_next == self.pos:
                # Swap — both move
                self.model.cell_occupant[self.pos] = occupant
                self.model.cell_occupant[target] = self
                occupant.pos = self.pos
                self.pos = target
                if self._path and self._path[0] == target:
                    self._path = self._path[1:]
                if occupant._path and occupant._path[0] == self.pos:
                    occupant._path = occupant._path[1:]
                self._record_move(target)
                return True
            return False  # genuinely blocked

        # Normal move
        self.model.cell_occupant.pop(self.pos, None)
        self.model.cell_occupant[target] = self
        self.pos = target
        if self._path and self._path[0] == target:
            self._path = self._path[1:]
        self._record_move(target)

        # Check if reached exit
        if self.model.env.grid[target[0], target[1]] == EXIT:
            self.state = "escaped"
            self.escape_step = self.model.schedule.steps
            self.model.register_escaped(self)
        return True

    def _find_alternative_step(self, blocked: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Find an adjacent free cell that reduces distance to target exit."""
        if self._target_exit is None:
            return None
        dm = self.model.exit_maps.maps.get(self._target_exit)
        if dm is None:
            return None
        r, c = self.pos
        tr, tc = self._target_exit
        best = None
        best_d = float("inf")
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if (nr, nc) == blocked:
                continue
            rows, cols = self.model.env.grid.shape
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = self.model.env.grid[nr, nc]
            if cell in (WALL, ALARM, FIRE):
                continue
            if self.model.cell_occupant.get((nr, nc)) is not None:
                continue
            d = dm.distance_from((nr, nc))
            if d < best_d:
                best_d = d
                best = (nr, nc)
        return best

    def _random_move(self) -> None:
        if not self._should_move_this_step():
            return
        nxt = random_walk(self.model.env.grid, self.pos, avoid_hazards=True)
        if nxt != self.pos and self.model.cell_occupant.get(nxt) is None:
            self.model.cell_occupant.pop(self.pos, None)
            self.model.cell_occupant[nxt] = self
            self.pos = nxt
            self._record_move(nxt)

    def _panic_move(self) -> None:
        if not self._should_move_this_step():
            return
        nxt = random_walk(self.model.env.grid, self.pos, avoid_hazards=False)
        if nxt != self.pos and self.model.cell_occupant.get(nxt) is None:
            self.model.cell_occupant.pop(self.pos, None)
            self.model.cell_occupant[nxt] = self
            self.pos = nxt
            self._record_move(nxt)
        # Periodically retry pathfinding from panic
        if self.model.schedule.steps % 5 == 0:
            self.state = "aware"

    def _record_move(self, new_pos: Tuple[int, int]) -> None:
        if self._last_pos is not None and self._last_pos != new_pos:
            self.cells_traversed += 1
        cell = self.model.env.grid[new_pos[0], new_pos[1]]
        if cell == SMOKE:
            self.steps_in_smoke += 1
            self.hazard_cells_crossed += 1
        elif cell == FIRE:
            self.steps_in_fire += 1
            self.hazard_cells_crossed += 1
        self._last_pos = new_pos

    # ------------------------------------------------------------------ #
    #  Health                                                              #
    # ------------------------------------------------------------------ #

    def _update_health(self, cell: int) -> None:
        if cell == SMOKE:
            self.health -= self.model.cfg.fire.smoke_damage
            self.steps_in_smoke += 1
        elif cell == FIRE:
            self.health -= self.model.cfg.fire.fire_damage
            self.steps_in_fire += 1
        if self.health <= 0:
            self.state = "dead"
            self.model.register_dead(self)

    # ------------------------------------------------------------------ #
    #  Family rescue                                                       #
    # ------------------------------------------------------------------ #

    def _find_relative_in_danger(self) -> Optional[int]:
        """Return unique_id of a nearby relative who is in danger."""
        for uid in self.relatives:
            rel = self.model.agent_by_id.get(uid)
            if rel is None or rel.state in ("escaped", "dead"):
                continue
            if rel.state in ("evacuating", "rescuing", "aware", "panic"):
                # Relative is already handling evacuation
                continue
            # A relative still in 'idle' when alarm is active needs help
            ar, ac = rel.pos
            r, c = self.pos
            if abs(ar - r) <= 12 and abs(ac - c) <= 12:
                return uid
        return None

    def _do_rescue(self) -> None:
        """Move toward relative, then escort together to exit."""
        self._rescue_steps += 1
        if self._rescue_steps > self.rescue_timeout:
            # Give up rescuing — self-evacuate
            self._rescue_target_id = None
            self.state = "aware"
            return

        rel = self.model.agent_by_id.get(self._rescue_target_id)
        if rel is None or rel.state in ("escaped", "dead"):
            # Relative no longer needs help
            self.rescue_completed = True
            self._rescue_target_id = None
            self.state = "aware"
            return

        # If we reached the relative, escort to exit together
        r, c = self.pos
        rr, rc = rel.pos
        if abs(r - rr) <= 1 and abs(c - rc) <= 1:
            # Both agents now share the same target and strategy
            rel.state = "aware"
            rel._target_exit = self._target_exit
            self.rescue_completed = True
            self._rescue_target_id = None
            self.state = "aware"
            return

        # Move toward relative via A*
        path = astar(self.model.env.grid, self.pos, rel.pos, risk_tolerance=self.risk)
        if path and len(path) > 1:
            self._try_move(path[1])
        else:
            # Can't reach — give up
            self._rescue_target_id = None
            self.state = "aware"
