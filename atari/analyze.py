"""
Analyze results.tsv and generate visualizations of the autonomous research run.

Usage:
    uv run analyze.py                # generate charts + print summary
    uv run analyze.py --no-show      # save PNG only, don't open window
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "run_analysis.png")


def load_results():
    experiments = []
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            experiments.append({
                "commit": row["commit"],
                "reward": float(row["mean_reward"]),
                "status": row["status"],
                "description": row.get("description", ""),
            })
    return experiments


def main():
    parser = argparse.ArgumentParser(description="Analyze autoresearch results")
    parser.add_argument("--no-show", action="store_true", help="Don't open plot window")
    args = parser.parse_args()

    experiments = load_results()
    if not experiments:
        print("No experiments found in results.tsv")
        return

    colors = {"keep": "#4CAF50", "discard": "#FF9800", "crash": "#f44336"}

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        f"AutoResearch Run Analysis — {len(experiments)} Experiments",
        fontsize=16, fontweight="bold", y=0.98,
    )

    # --- 1. All experiments scatter ---
    ax = axes[0][0]
    for i, e in enumerate(experiments):
        ax.scatter(i, e["reward"], c=colors[e["status"]], s=8, alpha=0.6, edgecolors="none")
    ax.set_title("All Experiments", fontweight="bold")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Mean Reward")
    legend_patches = [mpatches.Patch(color=c, label=s) for s, c in colors.items()]
    ax.legend(handles=legend_patches, fontsize=8)

    # --- 2. Kept experiments (monotonic improvement) ---
    ax = axes[0][1]
    kept_indices = []
    kept_rewards = []
    best_so_far = -1
    for i, e in enumerate(experiments):
        if e["status"] == "keep" and e["reward"] > best_so_far:
            kept_indices.append(i)
            kept_rewards.append(e["reward"])
            best_so_far = e["reward"]

    ax.plot(kept_indices, kept_rewards, "g-o", markersize=5, linewidth=2)
    for i, (idx, reward) in enumerate(zip(kept_indices, kept_rewards)):
        if reward > 20 or i == 0 or i == len(kept_indices) - 1:
            ax.annotate(f"{reward:.0f}", (idx, reward), textcoords="offset points",
                        xytext=(8, 5), fontsize=7, color="#2E7D32")
    ax.set_title("Reward Progression (Kept Only)", fontweight="bold")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Mean Reward")
    ax.set_yscale("log" if max(kept_rewards) / max(min(kept_rewards), 0.1) > 50 else "linear")

    # --- 3. Outcome distribution ---
    ax = axes[0][2]
    counts = {"keep": 0, "discard": 0, "crash": 0}
    for e in experiments:
        counts[e["status"]] += 1
    bars = ax.bar(counts.keys(), counts.values(), color=[colors[s] for s in counts])
    for bar, count in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{count}\n({count/len(experiments)*100:.0f}%)",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Outcome Distribution", fontweight="bold")
    ax.set_ylabel("Count")

    # --- 4. Crash rate over time (rolling window) ---
    ax = axes[1][0]
    window = min(20, len(experiments) // 4)
    if window >= 2:
        crash_rate = []
        for i in range(window, len(experiments)):
            batch = experiments[i - window:i]
            rate = sum(1 for e in batch if e["status"] == "crash") / window
            crash_rate.append(rate)
        ax.plot(range(window, len(experiments)), crash_rate, "r-", alpha=0.7, linewidth=1.5)
        ax.fill_between(range(window, len(experiments)), crash_rate, alpha=0.15, color="red")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
        ax.legend(fontsize=8)
    ax.set_title(f"Crash Rate (rolling {window}-experiment window)", fontweight="bold")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Crash Rate")
    ax.set_ylim(0, 1)

    # --- 5. Reward distribution for non-crash experiments ---
    ax = axes[1][1]
    non_crash_rewards = [e["reward"] for e in experiments if e["status"] != "crash" and e["reward"] > 0]
    if non_crash_rewards:
        ax.hist(non_crash_rewards, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(x=np.median(non_crash_rewards), color="red", linestyle="--", label=f"Median: {np.median(non_crash_rewards):.1f}")
        ax.axvline(x=max(non_crash_rewards), color="green", linestyle="--", label=f"Best: {max(non_crash_rewards):.1f}")
        ax.legend(fontsize=8)
    ax.set_title("Reward Distribution (non-crash)", fontweight="bold")
    ax.set_xlabel("Mean Reward")
    ax.set_ylabel("Count")

    # --- 6. Cumulative progress ---
    ax = axes[1][2]
    cumulative_best = []
    best = 0
    for e in experiments:
        if e["status"] == "keep" and e["reward"] > best:
            best = e["reward"]
        cumulative_best.append(best)
    ax.fill_between(range(len(experiments)), cumulative_best, alpha=0.3, color="#4CAF50")
    ax.plot(cumulative_best, color="#2E7D32", linewidth=2)
    ax.set_title("Cumulative Best Reward", fontweight="bold")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Best Reward So Far")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")

    # --- Print summary ---
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total experiments:  {len(experiments)}")
    print(f"  Kept:               {counts['keep']} ({counts['keep']/len(experiments)*100:.1f}%)")
    print(f"  Discarded:          {counts['discard']} ({counts['discard']/len(experiments)*100:.1f}%)")
    print(f"  Crashed:            {counts['crash']} ({counts['crash']/len(experiments)*100:.1f}%)")
    print(f"  Best reward:        {max(kept_rewards):.2f}")
    print(f"  Baseline:           {kept_rewards[0]:.2f}")
    print(f"  Improvement:        {max(kept_rewards)/kept_rewards[0]:.1f}x")
    print(f"{'=' * 60}")
    print(f"\n  Improvement milestones:")
    for idx, reward in zip(kept_indices, kept_rewards):
        print(f"    Experiment #{idx+1:>4}:  reward = {reward:>8.2f}")
    print(f"{'=' * 60}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
