"""
Plot experiment progression from results.tsv.
Shows val_nll and seq_recovery improving over successive runs.

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
        print("Run `uv run train.py` first to generate results.")
        sys.exit(1)

    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            row = dict(zip(header, vals))
            rows.append(row)

    if not rows:
        print("results.tsv is empty.")
        sys.exit(1)

    return rows


def plot(rows, save=False):
    runs = list(range(1, len(rows) + 1))
    val_nll = [float(r["val_nll"]) for r in rows]
    seq_recovery = [float(r["seq_recovery"]) * 100 for r in rows]
    perplexity = [float(r["val_perplexity"]) for r in rows]

    # Track best val_nll so far
    best_nll = []
    current_best = float("inf")
    for nll in val_nll:
        current_best = min(current_best, nll)
        best_nll.append(current_best)

    if save:
        matplotlib.use("Agg")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # val_nll with best-so-far envelope
    axes[0].plot(runs, val_nll, "o-", color="#2196F3", alpha=0.6, label="val_nll")
    axes[0].plot(runs, best_nll, "-", color="#F44336", linewidth=2, label="best so far")
    axes[0].set_xlabel("Experiment #")
    axes[0].set_ylabel("val_nll (lower is better)")
    axes[0].set_title("Validation NLL")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Perplexity
    axes[1].plot(runs, perplexity, "s-", color="#FF9800")
    axes[1].set_xlabel("Experiment #")
    axes[1].set_ylabel("Perplexity (lower is better)")
    axes[1].set_title("Validation Perplexity")
    axes[1].grid(True, alpha=0.3)

    # Sequence recovery
    axes[2].plot(runs, seq_recovery, "^-", color="#4CAF50")
    axes[2].set_xlabel("Experiment #")
    axes[2].set_ylabel("Seq Recovery % (higher is better)")
    axes[2].set_title("Sequence Recovery")
    axes[2].grid(True, alpha=0.3)

    # Annotate each point with key hyperparams
    for i, r in enumerate(rows):
        label = f"d{r.get('d_model', '?')}_L{r.get('depth', '?')}"
        axes[0].annotate(label, (runs[i], val_nll[i]),
                         textcoords="offset points", xytext=(0, 8),
                         fontsize=6, ha="center", alpha=0.7)

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
