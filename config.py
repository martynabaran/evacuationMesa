"""
Configuration dataclasses for the Mesa evacuation simulation.
Replaces the split settings.py / config.yaml approach from the original.
"""
from dataclasses import dataclass, field
from typing import Tuple, Dict
import yaml


@dataclass
class AgentConfig:
    health: int = 100
    risk_range: Tuple[float, float] = (0.1, 0.9)
    perception_range: int = 5
    communication_range: int = 5
    # Strategy distribution — keys must sum to num_agents
    strategies: Dict[str, int] = field(default_factory=lambda: {
        "nearest_exit": 10,
        "safest_exit": 10,
        "least_crowded_exit": 10,
    })


@dataclass
class FamilyConfig:
    num_families: int = 8
    min_size: int = 2
    max_size: int = 3
    # Timeout steps before a waiting agent abandons rescue and self-evacuates
    rescue_timeout: int = 30


@dataclass
class FireConfig:
    # Probability that fire spreads to an adjacent cell per propagation step
    fire_spread_prob: float = 0.6
    # Probability that smoke spreads to an adjacent cell per propagation step
    smoke_spread_prob: float = 0.3
    # Fire propagation interval (model steps)
    fire_interval: int = 5
    # Smoke propagation interval (model steps)
    smoke_interval: int = 4
    # Damage per step in smoke / fire
    smoke_damage: int = 2
    fire_damage: int = 50


@dataclass
class SimulationConfig:
    layout_path: str = "room_layouts/supermarket3.txt"
    num_agents: int = 30
    num_communicative: int = 30
    seed: int = 42
    max_steps: int = 2000
    agent: AgentConfig = field(default_factory=AgentConfig)
    family: FamilyConfig = field(default_factory=FamilyConfig)
    fire: FireConfig = field(default_factory=FireConfig)
    # Age range affects speed: linear interpolation between speed_old and speed_young
    age_min: int = 18
    age_max: int = 80
    speed_young: float = 1.0    # cells/step for age == age_min
    speed_old: float = 0.5      # cells/step for age == age_max


def load_config(path: str = "config.yaml") -> SimulationConfig:
    """Load configuration from YAML, falling back to defaults for missing keys."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raw = {}

    sim = raw.get("simulation", {})
    agent_attrs = sim.get("agent_attributes", {})
    strats = sim.get("strategies", {})
    fire_cfg = sim.get("fire", {})
    family_cfg = sim.get("family", {})

    return SimulationConfig(
        layout_path=agent_attrs.get("layout", "room_layouts/supermarket3.txt").replace("\\", "/"),
        num_agents=sim.get("num_agents", 30),
        num_communicative=agent_attrs.get("communicates", 30),
        seed=sim.get("seed", 42),
        max_steps=sim.get("max_steps", 2000),
        agent=AgentConfig(
            health=agent_attrs.get("health", 100),
            risk_range=tuple(agent_attrs.get("risk", [0.1, 0.9])),
            perception_range=agent_attrs.get("range", 5),
            communication_range=agent_attrs.get("volume", 5),
            strategies=strats if strats else {"nearest_exit": 10, "safest_exit": 10, "least_crowded_exit": 10},
        ),
        family=FamilyConfig(
            num_families=sim.get("num_families", 8),
            rescue_timeout=sim.get("rescue_timeout", 30),
        ),
        fire=FireConfig(
            fire_spread_prob=sim.get("fire_probability", 0.6),
            smoke_spread_prob=sim.get("smoke_probability", 0.3),
            smoke_damage=sim.get("smoke_damage", 2),
            fire_damage=sim.get("fire_damage", 50),
        ),
    )
