# karprotein

Autonomous research for protein inverse folding, built on the [autoresearch](https://github.com/karpathy/autoresearch) workflow.

The idea: give an AI agent a protein inverse folding training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

## How it works

The repo is deliberately kept small and only has three files that matter:

- **`prepare.py`** -- fixed constants, one-time data prep (downloads CATH 4.2 dataset), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** -- the single file the agent edits. Contains the protein transformer model, optimizer, and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, feature engineering, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** -- baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_nll** (validation negative log-likelihood per residue) -- lower is better. Secondary metrics include sequence recovery (% correct amino acid predictions) and perplexity.

## The problem

**Protein inverse folding**: given a protein's 3D backbone structure (N, CA, C, O atom coordinates for each residue), predict the amino acid sequence that would fold into that structure. This is a 20-class classification problem at each residue position, where the model must learn the relationship between local 3D geometry and amino acid preferences.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download and process CATH 4.2 dataset (one-time)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      -- constants, data prep + runtime utilities (do not modify)
train.py        -- model, optimizer, training loop (agent modifies this)
program.md      -- agent instructions
pyproject.toml  -- dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc).
- **Self-contained.** No external dependencies beyond PyTorch, numpy, and requests. One GPU, one file, one metric.

## Dataset

The CATH 4.2 inverse-folding dataset, originally released with the graph-protein-design work. Contains ~18K training proteins, ~2K validation, ~1.1K test. Each protein has backbone atom coordinates (N, CA, C, O) and the corresponding amino acid sequence.

## License

MIT
