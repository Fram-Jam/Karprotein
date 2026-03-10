# karprotein

Autonomous research for protein inverse folding: given a protein's 3D backbone structure (N, CA, C, O atom coordinates), predict the amino acid sequence that folds into that structure.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `prepare.py` -- fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` -- the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/karprotein/` contains processed data. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` -- this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, feature engineering, geometric representations, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `run_validation` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_nll (validation negative log-likelihood per residue).** Since the time budget is fixed, you don't need to worry about training time -- it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, the geometric feature extraction. The only constraint is that the code runs without crashing and finishes within the time budget.

**Secondary metrics:** `seq_recovery` (fraction of correctly predicted amino acids) and `val_perplexity` (exp of val_nll) are reported but val_nll is the primary optimization target.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_nll gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Domain context

Protein inverse folding is a structure-to-sequence prediction problem. Key facts:

- **Input**: backbone atom coordinates (N, CA, C, O) for each residue, shape (B, L, 4, 3)
- **Output**: amino acid identity for each residue, 20 classes (ACDEFGHIKLMNPQRSTVWY)
- **Key challenge**: the model must learn the relationship between local 3D geometry and amino acid preferences
- **What matters**: local geometric features (bond angles, distances, orientations), and longer-range structural context (secondary structure, packing interactions)
- **State of the art approaches** use geometric graph neural networks, invariant point attention, SE(3)-equivariant networks, or structure-aware transformers. The baseline uses a simple transformer encoder with hand-crafted geometric features.
- **Dataset**: CATH 4.2, ~18K training proteins, ~2K validation, ~1.1K test

## Output format

Once the script finishes it prints a summary like this:

```
---
val_nll:          2.853000
val_perplexity:   17.340000
seq_recovery:     0.332000
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     4500.2
num_steps:        9530
num_params_M:     5.3
depth:            6
```

You can extract the key metric from the log file:

```
grep "^val_nll:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated -- commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_nll	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_nll achieved (e.g. 2.853000) -- use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 4.4 -- divide peak_vram_mb by 1024) -- use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_nll	memory_gb	status	description
a1b2c3d	2.853000	4.4	keep	baseline
b2c3d4e	2.810000	4.5	keep	increase d_model to 384
c3d4e5f	2.870000	4.4	discard	switch to ReLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10` or `autoresearch/mar10-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything -- do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_nll:\|^seq_recovery:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_nll improved (lower), you "advance" the branch, keeping the git commit
9. If val_nll is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder -- read papers on protein inverse folding (ProteinMPNN, ESM-IF, GVP, IPA), re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
