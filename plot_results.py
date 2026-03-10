"""
Plot experiment progression from results.tsv.
Reads the program.md format: commit, val_nll, memory_gb, status, description.

Usage: uv run plot_results.py
       uv run plot_results.py --save  (saves to results.png instead of displaying)
"""

import subprocess
import sys

# matplotlib is optional -- install if missing
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    print("Installing matplotlib...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib
    import matplotlib.pyplot as plt

import os


def load_results(path="results.tsv"):
    if not os.path.exists(path):
        print(f"No results file found at {path}")
        print("Run experiments first to generate results.")
        sys.exit(1)

    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) < 4:
                continue
            row = dict(zip(header, vals))
            # Skip crashed runs (val_nll == 0)
            if row.get("status") == "crash":
                continue
            rows.append(row)

    if not rows:
        print("results.tsv has no successful runs.")
        sys.exit(1)

    return rows


def plot(rows, save=False):
    runs = list(range(1, len(rows) + 1))
    val_nll = [float(r["val_nll"]) for r in rows]
    memory_gb = [float(r["memory_gb"]) for r in rows]
    descriptions = [r.get("description", "") for r in rows]

    # Track best val_nll so far
    best_nll = []
    current_best = float("inf")
    for nll in val_nll:
        current_best = min(current_best, nll)
        best_nll.append(current_best)

    if save:
        matplotlib.use("Agg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # val_nll with best-so-far envelope
    axes[0].plot(runs, val_nll, "o-", color="#2196F3", alpha=0.6, label="val_nll")
    axes[0].plot(runs, best_nll, "-", color="#F44336", linewidth=2, label="best so far")
    axes[0].set_xlabel("Experiment #")
    axes[0].set_ylabel("val_nll (lower is better)")
    axes[0].set_title("Validation NLL")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Memory usage
    axes[1].bar(runs, memory_gb, color="#FF9800", alpha=0.7)
    axes[1].set_xlabel("Experiment #")
    axes[1].set_ylabel("Peak VRAM (GB)")
    axes[1].set_title("Memory Usage")
    axes[1].grid(True, alpha=0.3)

    # Annotate each point with description
    for i, desc in enumerate(descriptions):
        short = desc[:20] if len(desc) > 20 else desc
        axes[0].annotate(short, (runs[i], val_nll[i]),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=6, ha="center", alpha=0.7, rotation=30)

    plt.tight_layout()

    if save:
        out = "results.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    rows = load_results()
    save = "--save" in sys.argv
    print(f"Loaded {len(rows)} experiments from results.tsv")
    plot(rows, save=save)
