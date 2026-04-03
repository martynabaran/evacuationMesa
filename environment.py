"""
Grid environment, fire/smoke cellular automaton, and precomputed exit distance maps.

Key optimization over the original:
- Single shared numpy grid (not a deep copy per agent).
- Fire/smoke propagation uses numpy boolean operations — no mid-iteration mutation.
- Exit distance maps are computed ONCE per exit via reverse-BFS/Dijkstra and
  reused by every agent. They are invalidated and lazily recomputed only when
  fire/smoke cells change near exit corridors.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Cell type constants (integer grid — faster than character comparisons)
OPEN  = 0
WALL  = 1
EXIT  = 2
FIRE  = 3
SMOKE = 4
ALARM = 5

CHAR_MAP = {"O": OPEN, "W": WALL, "E": EXIT, "F": FIRE, "S": SMOKE, "A": ALARM}
INT_MAP  = {v: k for k, v in CHAR_MAP.items()}

# 4-directional movement deltas
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def load_layout(path: str) -> np.ndarray:
    """Parse a text layout file into an integer numpy grid (rows × cols)."""
    with open(path) as f:
        rows = [line.split() for line in f.read().splitlines() if line.strip()]
    height = len(rows)
    width  = max(len(r) for r in rows)
    grid = np.zeros((height, width), dtype=np.int8)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            grid[r, c] = CHAR_MAP.get(ch, OPEN)
    return grid


def get_exits(grid: np.ndarray) -> List[Tuple[int, int]]:
    positions = list(zip(*np.where(grid == EXIT)))
    return [(int(r), int(c)) for r, c in positions]


def get_alarms(grid: np.ndarray) -> List[Tuple[int, int]]:
    positions = list(zip(*np.where(grid == ALARM)))
    return [(int(r), int(c)) for r, c in positions]


class DistanceMap:
    """
    Precomputed distance map from every passable cell to a single exit,
    weighted by smoke (risk-neutral weight) and fire (impassable).

    Recomputed whenever the hazard configuration changes enough to
    potentially alter routing (see `needs_recompute`).
    """

    def __init__(self, grid: np.ndarray, exit_pos: Tuple[int, int], smoke_weight: float = 1.5):
        self.exit_pos = exit_pos
        self.smoke_weight = smoke_weight
        self._dist: np.ndarray = np.full(grid.shape, np.inf)
        self._parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        self._last_hazard_hash: int = 0
        self.recompute(grid)

    def recompute(self, grid: np.ndarray) -> None:
        """
        Reverse-Dijkstra from exit_pos.
        Running once per exit serves ALL agents — O(V log V) shared cost.
        """
        rows, cols = grid.shape
        dist = np.full((rows, cols), np.inf)
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        er, ec = self.exit_pos

        dist[er, ec] = 0.0
        parent[(er, ec)] = None
        heap = [(0.0, er, ec)]

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                cell = grid[nr, nc]
                if cell == WALL or cell == ALARM:
                    continue
                if cell == FIRE:
                    # Fire cells are impassable in the base distance map.
                    # Agents with high risk tolerance can override this in A*.
                    continue
                weight = 1.0
                if cell == SMOKE:
                    weight += self.smoke_weight
                nd = d + weight
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    parent[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (nd, nr, nc))

        self._dist = dist
        self._parent = parent
        self._last_hazard_hash = self._hazard_hash(grid)

    def _hazard_hash(self, grid: np.ndarray) -> int:
        """Cheap hash of hazard cells to detect when recomputation is needed."""
        hazard = (grid == FIRE) | (grid == SMOKE)
        return hash(hazard.tobytes())

    def needs_recompute(self, grid: np.ndarray) -> bool:
        return self._hazard_hash(grid) != self._last_hazard_hash

    def distance_from(self, pos: Tuple[int, int]) -> float:
        return float(self._dist[pos[0], pos[1]])

    def next_step(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Returns the next cell toward this exit along the shortest path,
        or None if unreachable.
        Reconstructs one hop from the parent map — no full path storage needed.
        """
        r, c = pos
        if np.isinf(self._dist[r, c]):
            return None
        er, ec = self.exit_pos
        # Walk parent pointers until we find a cell whose parent is pos.
        # For large grids, cache the forward direction lazily.
        # Here we use the gradient: find the neighbor with the smallest distance.
        best = None
        best_d = self._dist[r, c]
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            rows, cols = self._dist.shape
            if 0 <= nr < rows and 0 <= nc < cols:
                nd = self._dist[nr, nc]
                if nd < best_d:
                    best_d = nd
                    best = (nr, nc)
        return best


class ExitDistanceMaps:
    """
    Container for all per-exit DistanceMaps.
    Lazily recomputes maps when fire/smoke changes.
    """

    def __init__(self, grid: np.ndarray, exits: List[Tuple[int, int]]):
        self._grid_ref = grid
        self.exits = exits
        self.maps: Dict[Tuple[int, int], DistanceMap] = {
            e: DistanceMap(grid, e) for e in exits
        }

    def maybe_recompute(self, grid: np.ndarray) -> None:
        """Call once per propagation step (not once per agent)."""
        for exit_pos, dm in self.maps.items():
            if dm.needs_recompute(grid):
                dm.recompute(grid)

    def best_exit_for(
        self,
        pos: Tuple[int, int],
        strategy: str,
        agent_risk: float,
        agent_grid: np.ndarray,
        all_agent_positions: Set[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """
        Select the target exit based on strategy.
        - nearest_exit:      minimum distance from current position
        - safest_exit:       minimum hazard along path (penalty for smoke/fire density)
        - least_crowded_exit: minimum (distance + crowding penalty)
        """
        valid = [e for e in self.exits if not np.isinf(self.maps[e].distance_from(pos))]
        if not valid:
            return self.exits[0]  # fallback

        if strategy == "nearest_exit":
            return min(valid, key=lambda e: self.maps[e].distance_from(pos))

        elif strategy == "safest_exit":
            def safety_score(exit_pos):
                # Penalize exits surrounded by fire/smoke within radius 5
                r, c = exit_pos
                rows, cols = agent_grid.shape
                hazard_count = 0
                for dr in range(-5, 6):
                    for dc in range(-5, 6):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if agent_grid[nr, nc] in (FIRE, SMOKE):
                                hazard_count += 1
                return self.maps[exit_pos].distance_from(pos) + hazard_count * 2.0
            return min(valid, key=safety_score)

        elif strategy == "least_crowded_exit":
            def crowding_score(exit_pos):
                r, c = exit_pos
                crowding = sum(
                    1 for (ar, ac) in all_agent_positions
                    if abs(ar - r) <= 8 and abs(ac - c) <= 8
                )
                return self.maps[exit_pos].distance_from(pos) + crowding * 3.0
            return min(valid, key=crowding_score)

        return min(valid, key=lambda e: self.maps[e].distance_from(pos))


class FireEnvironment:
    """
    Cellular automaton for fire and smoke propagation.
    Uses numpy boolean masking instead of per-cell Python loops —
    avoids the mid-iteration mutation bug of the original.
    """

    def __init__(self, grid: np.ndarray, rng: np.random.Generator):
        self.grid = grid  # shared reference — agents read this directly
        self.rng = rng
        self._rows, self._cols = grid.shape

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _candidate_mask(self, source_type: int) -> np.ndarray:
        """Boolean mask of cells adjacent to `source_type` that can receive spread."""
        source = self.grid == source_type
        passable = (
            (self.grid == OPEN) | (self.grid == SMOKE)
        )
        # Shift source mask in all 4 directions and OR together
        spread = (
            np.roll(source, 1, axis=0) |
            np.roll(source, -1, axis=0) |
            np.roll(source, 1, axis=1) |
            np.roll(source, -1, axis=1)
        )
        # Edges wrapped by roll — zero them out
        spread[0, :] = False
        spread[-1, :] = False
        spread[:, 0] = False
        spread[:, -1] = False
        return spread & passable

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def propagate_fire(self, prob: float) -> Set[Tuple[int, int]]:
        """
        Spread fire to candidate adjacent cells with probability `prob`.
        Returns set of newly ignited cells.
        """
        candidates = self._candidate_mask(FIRE)
        rand = self.rng.random(self.grid.shape)
        new_fire = candidates & (rand < prob)
        # Convert smoke→fire where fire spreads into smoke
        coords = list(zip(*np.where(new_fire)))
        for r, c in coords:
            self.grid[r, c] = FIRE
        return {(int(r), int(c)) for r, c in coords}

    def propagate_smoke(self, prob: float) -> Set[Tuple[int, int]]:
        """
        Spread smoke from both fire cells and existing smoke cells.
        Newly ignited fire cells from this step do not generate smoke yet.
        Returns set of newly smoked cells.
        """
        # Smoke spreads from fire and existing smoke
        from_fire  = self._candidate_mask(FIRE)
        from_smoke = self._candidate_mask(SMOKE)
        candidates = from_fire | from_smoke

        rand = self.rng.random(self.grid.shape)
        new_smoke = candidates & (rand < prob) & (self.grid == OPEN)
        coords = list(zip(*np.where(new_smoke)))
        for r, c in coords:
            self.grid[r, c] = SMOKE
        return {(int(r), int(c)) for r, c in coords}

    def ignite(self, r: int, c: int) -> None:
        """Place initial fire at (r, c)."""
        if self.grid[r, c] not in (WALL, EXIT, ALARM):
            self.grid[r, c] = FIRE

    def check_alarms(self, alarm_positions: List[Tuple[int, int]], radius: int = 1) -> bool:
        """Return True if any alarm detects fire or smoke within radius."""
        rows, cols = self.grid.shape
        for ar, ac in alarm_positions:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = ar + dr, ac + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if self.grid[nr, nc] in (FIRE, SMOKE):
                            return True
        return False
