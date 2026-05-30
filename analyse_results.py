"""
analyse_results.py — Run the evacuation simulation and produce all analysis
outputs used in the final report.

Steps performed:
  1. Single reference run (seed=42) → results/model_timeseries.csv
                                     results/agent_data.csv
  2. Batch run (100 seeds)          → results/batch_results.csv
  3. Statistical summaries          → printed to stdout
  4. Figures                        → results/analysis_plots.png
                                     results/timeseries_plot.png

Usage:
    python analyse_results.py              # full run (default 100 replications)
    python analyse_results.py --n-reps 20  # quick smoke-test
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config, SimulationConfig
from model import EvacuationModel


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

OUTPUT_DIR = "results"


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


# ------------------------------------------------------------------ #
#  Step 1 – Single reference run                                      #
# ------------------------------------------------------------------ #

def run_reference(seed: int = 42) -> EvacuationModel:
    """Run a single deterministic simulation and save per-step timeseries."""
    print(f"\n[1/4] Reference run (seed={seed}) …")
    cfg = load_config()
    cfg = SimulationConfig(
        layout_path=cfg.layout_path,
        num_agents=cfg.num_agents,
        seed=seed,
        agent=cfg.agent,
        family=cfg.family,
        fire=cfg.fire,
    )
    model = EvacuationModel(cfg)
    while model.running:
        model.step()

    model.export_results(OUTPUT_DIR)

    total  = cfg.num_agents
    esc    = len(model.escaped_agents)
    dead   = len(model.dead_agents)
    steps  = model.schedule.steps
    print(f"    Finished in {steps} steps — "
          f"escaped={esc}/{total} ({100*esc/total:.1f}%), "
          f"dead={dead}/{total} ({100*dead/total:.1f}%)")
    return model


# ------------------------------------------------------------------ #
#  Step 2 – Batch run                                                 #
# ------------------------------------------------------------------ #

def run_batch(n_reps: int = 100) -> pd.DataFrame:
    """Run n_reps replications (seed 0 … n_reps-1) and save batch CSV."""
    print(f"\n[2/4] Batch run ({n_reps} replications) …")
    from mesa.batchrunner import batch_run

    cfg = load_config()
    parameters = {
        "cfg": [
            SimulationConfig(
                layout_path=cfg.layout_path,
                num_agents=cfg.num_agents,
                seed=seed,
                agent=cfg.agent,
                family=cfg.family,
                fire=cfg.fire,
            )
            for seed in range(n_reps)
        ]
    }

    results = batch_run(
        EvacuationModel,
        parameters=parameters,
        iterations=1,
        max_steps=cfg.max_steps,
        data_collection_period=-1,   # collect only at end of each run
        display_progress=True,
    )

    df = pd.DataFrame(results)
    out = _path("batch_results.csv")
    df.to_csv(out, index=False)
    print(f"    Saved → {out}")
    return df


# ------------------------------------------------------------------ #
#  Step 3 – Statistical summaries                                     #
# ------------------------------------------------------------------ #

def print_statistics(df: pd.DataFrame) -> None:
    """Print all summary statistics cited in the final report."""
    print("\n[3/4] Statistical summaries")
    print("=" * 60)

    df["survived"] = (df["State"] == "escaped").astype(int)
    runs = df.groupby("RunId").first()

    # --- Overall survival ---
    print("\n--- Overall survival (per-run) ---")
    desc = runs[["Escaped", "Dead", "EvacuationRate", "Step"]].describe()
    print(desc.to_string())
    print(f"\n  Mean survival rate : {runs.EvacuationRate.mean()*100:.1f}%  "
          f"(±{runs.EvacuationRate.std()*100:.1f}%)")
    print(f"  Min / Max          : {runs.EvacuationRate.min()*100:.1f}% / "
          f"{runs.EvacuationRate.max()*100:.1f}%")
    print(f"  Mean sim duration  : {runs.Step.mean():.1f} steps  "
          f"(range {runs.Step.min()}–{runs.Step.max()})")

    # --- Strategy ---
    print("\n--- Survival rate by strategy ---")
    for s in ["nearest_exit", "safest_exit", "least_crowded_exit"]:
        sub = df[df["Strategy"] == s]
        esc  = (sub["State"] == "escaped").sum()
        dead = (sub["State"] == "dead").sum()
        n    = len(sub)
        print(f"  {s:26s}  escaped={esc:4d} ({100*esc/n:.1f}%)  "
              f"dead={dead:3d} ({100*dead/n:.1f}%)")

    escaped = df[df["State"] == "escaped"]
    print("\n--- Mean escape step by strategy ---")
    print(escaped.groupby("Strategy")["EscapeStep"].mean().round(1).to_string())

    # --- Age groups ---
    print("\n--- Survival rate by age group ---")
    bins   = [18, 30, 45, 60, 80]
    labels = ["18–30", "31–45", "46–60", "61–80"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)
    print(df.groupby("age_group", observed=False)["survived"]
            .mean().mul(100).round(1).to_string())

    # --- Escape time ---
    print("\n--- Escape step distribution (all escaped agents) ---")
    print(escaped["EscapeStep"].describe().round(1).to_string())

    # --- Communication ---
    print("\n--- Communication effect ---")
    df["got_comms"] = df["CommsReceived"] > 0
    for flag, label in [(False, "No comms received"), (True, "Comms received")]:
        rate = df[df["got_comms"] == flag]["survived"].mean() * 100
        print(f"  {label:22s}  survival={rate:.1f}%")
    print(f"\n  Mean CommsReceived — escaped : "
          f"{df[df['State']=='escaped']['CommsReceived'].mean():.1f}")
    print(f"  Mean CommsReceived — dead    : "
          f"{df[df['State']=='dead']['CommsReceived'].mean():.1f}")

    # --- Family ---
    print("\n--- Family membership effect ---")
    df["in_family"] = df["FamilyID"].notna()
    print(df.groupby("in_family")["survived"]
            .mean().mul(100).round(1).to_string())
    rescued = df[df["in_family"]]["RescueCompleted"]
    print(f"  Rescue completion rate : {rescued.sum()} / {len(rescued)} "
          f"({100*rescued.mean():.1f}%)")

    # --- Environmental spread ---
    print("\n--- Fire / smoke cells at simulation end ---")
    print(runs[["FireCells", "SmokeCells"]].describe().round(1).to_string())

    print("\n" + "=" * 60)


# ------------------------------------------------------------------ #
#  Step 4a – Six-panel analysis figure                                #
# ------------------------------------------------------------------ #

def plot_analysis(df: pd.DataFrame) -> None:
    """Six-panel figure covering all major analysis dimensions."""
    print("\n[4/4] Generating figures …")

    df["survived"] = (df["State"] == "escaped").astype(int)
    runs = df.groupby("RunId").first()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Evacuation Simulation — Analysis Results (100 runs)",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: survival rate histogram ────────────────────────────
    ax = axes[0, 0]
    ax.hist(runs["EvacuationRate"] * 100, bins=12,
            color="steelblue", edgecolor="white", linewidth=0.8)
    mean_rate = runs["EvacuationRate"].mean() * 100
    ax.axvline(mean_rate, color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean_rate:.1f}%")
    ax.set_xlabel("Survival Rate (%)")
    ax.set_ylabel("Number of Runs")
    ax.set_title("Distribution of Survival Rate\nAcross 100 Runs")
    ax.legend()

    # ── Panel 2: strategy comparison ────────────────────────────────
    ax = axes[0, 1]
    strats  = ["nearest_exit", "safest_exit", "least_crowded_exit"]
    labels  = ["Nearest Exit", "Safest Exit", "Least Crowded"]
    colors  = ["#e74c3c", "#2ecc71", "#3498db"]
    surv    = [(df[df["Strategy"] == s]["State"] == "escaped").mean() * 100
               for s in strats]
    dead_r  = [(df[df["Strategy"] == s]["State"] == "dead").mean() * 100
               for s in strats]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w / 2, surv,   w, label="Escaped", color=colors)
    ax.bar(        x + w / 2, dead_r, w, label="Dead",
           color=[c for c in ["#c0392b", "#27ae60", "#2980b9"]], alpha=0.6)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Survival Rate by\nEvacuation Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=8)

    # ── Panel 3: age group survival ─────────────────────────────────
    ax = axes[0, 2]
    bins_age   = [18, 30, 45, 60, 80]
    labels_age = ["18–30", "31–45", "46–60", "61–80"]
    df["age_group"] = pd.cut(df["Age"], bins=bins_age, labels=labels_age)
    age_surv = df.groupby("age_group", observed=False)["survived"].mean() * 100
    age_surv.plot(kind="bar", ax=ax,
                  color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
                  edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_title("Survival Rate by Age Group")
    ax.set_xticklabels(labels_age, rotation=0)
    ax.set_ylim(80, 100)
    for i, v in enumerate(age_surv):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    # ── Panel 4: escape step distribution ───────────────────────────
    ax = axes[1, 0]
    esc_df = df[df["State"] == "escaped"]
    ax.hist(esc_df["EscapeStep"], bins=30,
            color="#2ecc71", edgecolor="white", linewidth=0.8)
    mean_e   = esc_df["EscapeStep"].mean()
    median_e = esc_df["EscapeStep"].median()
    ax.axvline(mean_e,   color="red",    linestyle="--", linewidth=2,
               label=f"Mean: {mean_e:.1f}")
    ax.axvline(median_e, color="orange", linestyle="--", linewidth=2,
               label=f"Median: {median_e:.1f}")
    ax.set_xlabel("Escape Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Distribution of Individual\nEscape Times")
    ax.legend()

    # ── Panel 5: fire vs smoke scatter, coloured by survival rate ───
    ax = axes[1, 1]
    sc = ax.scatter(
        runs["FireCells"], runs["SmokeCells"],
        c=runs["EvacuationRate"] * 100,
        cmap="RdYlGn", alpha=0.8, edgecolor="grey", linewidth=0.3,
    )
    plt.colorbar(sc, ax=ax, label="Survival Rate (%)")
    ax.set_xlabel("Fire Cells at Simulation End")
    ax.set_ylabel("Smoke Cells at Simulation End")
    ax.set_title("Fire vs Smoke Spread\nat Simulation End")

    # ── Panel 6: communication effect ───────────────────────────────
    ax = axes[1, 2]
    df["got_comms"] = df["CommsReceived"] > 0
    comm_vals = [
        df[df["got_comms"] == False]["survived"].mean() * 100,
        df[df["got_comms"] == True ]["survived"].mean() * 100,
    ]
    bars = ax.bar(["No Comms\nReceived", "Comms\nReceived"],
                  comm_vals,
                  color=["#e74c3c", "#3498db"],
                  edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Survival Rate (%)")
    ax.set_title("Impact of Communication\non Survival")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, comm_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out = _path("analysis_plots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ------------------------------------------------------------------ #
#  Step 4b – Single-run timeseries figure                             #
# ------------------------------------------------------------------ #

def plot_timeseries() -> None:
    """Two-panel time-series figure from the reference run."""
    csv_path = _path("model_timeseries.csv")
    if not os.path.exists(csv_path):
        print(f"    [skip] {csv_path} not found — run reference simulation first.")
        return

    mdf = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Single-Run Simulation Dynamics (seed=42)",
        fontsize=13, fontweight="bold",
    )

    # Left: agent states
    ax = axes[0]
    ax.plot(mdf["Escaped"],  color="#2ecc71", linewidth=2.5, label="Escaped")
    ax.plot(mdf["Dead"],     color="#e74c3c", linewidth=2.5, label="Dead")
    ax.plot(mdf["Alive"],    color="#3498db", linewidth=2.5, label="Alive")
    ax.plot(mdf["Panicking"],color="#f39c12", linewidth=1.5, linestyle=":",  label="Panicking")
    ax.plot(mdf["Rescuing"], color="#9b59b6", linewidth=1.5, linestyle="--", label="Rescuing")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Agent Population States Over Time")
    ax.legend(loc="center right")
    ax.grid(alpha=0.3)

    # Right: environmental hazards + mean health
    ax2 = axes[1]
    l1, = ax2.plot(mdf["FireCells"],  color="#e74c3c", linewidth=2,   label="Fire Cells")
    l2, = ax2.plot(mdf["SmokeCells"], color="#95a5a6", linewidth=2,   linestyle="--", label="Smoke Cells")
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Number of Cells")
    ax2_r = ax2.twinx()
    l3, = ax2_r.plot(mdf["MeanHealth"], color="#27ae60", linewidth=2.5,
                     linestyle="-.", label="Mean Agent Health")
    ax2_r.set_ylabel("Mean Health (HP)", color="#27ae60")
    ax2_r.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_title("Environmental Hazards and Agent Health")
    ax2.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = _path("timeseries_plot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {out}")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evacuation simulation and produce report analysis outputs."
    )
    parser.add_argument(
        "--n-reps", type=int, default=100,
        help="Number of batch replications (default: 100)",
    )
    parser.add_argument(
        "--skip-batch", action="store_true",
        help="Skip batch run and use existing batch_results.csv",
    )
    args = parser.parse_args()

    _ensure_output_dir()

    # 1. Reference run
    run_reference(seed=42)

    # 2. Batch run
    batch_csv = _path("batch_results.csv")
    if args.skip_batch and os.path.exists(batch_csv):
        print(f"\n[2/4] Skipping batch run — loading {batch_csv}")
        df = pd.read_csv(batch_csv)
    else:
        df = run_batch(n_reps=args.n_reps)

    # 3. Statistics
    print_statistics(df)

    # 4. Figures
    plot_analysis(df)
    plot_timeseries()

    print("\nDone. All outputs written to the results/ directory.")


if __name__ == "__main__":
    main()
