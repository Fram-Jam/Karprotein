"""
One-time data preparation and fixed runtime utilities for protein inverse folding.

Usage:
    uv run prepare.py

Downloads the CATH 4.2 inverse-folding dataset, preprocesses each protein into
a cached tensor representation, and exposes the fixed dataloading and evaluation
utilities used by train.py.

Data is stored in ~/.cache/karprotein/.
"""

import os
import sys
import time
import json
import math
from dataclasses import dataclass

import numpy as np
import requests
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
NUM_CLASSES = len(AA_ALPHABET)  # 20
BACKBONE_ATOMS = ("N", "CA", "C", "O")

MAX_SEQ_LEN = 1024       # maximum protein sequence length
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_RESIDUES = 100_000   # number of residues for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "karprotein")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
PROCESSED_DIR = os.path.join(CACHE_DIR, "processed")

CHAIN_SET_URL = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl"
SPLITS_URL = "https://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json"

CHAIN_SET_PATH = os.path.join(RAW_DIR, "chain_set.jsonl")
SPLITS_PATH = os.path.join(RAW_DIR, "chain_set_splits.json")

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_file(url, destination):
    """Download a file with retries. Returns True on success."""
    if os.path.exists(destination):
        return True

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = destination + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, destination)
            print(f"  Downloaded {os.path.basename(destination)}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
            for path in [destination + ".tmp", destination]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data():
    """Download CATH 4.2 dataset files."""
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(CHAIN_SET_PATH) and os.path.exists(SPLITS_PATH):
        print(f"Data: already downloaded at {RAW_DIR}")
        return

    print("Data: downloading CATH 4.2 dataset...")
    ok1 = _download_file(CHAIN_SET_URL, CHAIN_SET_PATH)
    ok2 = _download_file(SPLITS_URL, SPLITS_PATH)
    if not (ok1 and ok2):
        print("Data: download failed. Check your internet connection.")
        sys.exit(1)
    print(f"Data: ready at {RAW_DIR}")

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------

def _load_split_names():
    with open(SPLITS_PATH) as f:
        raw = json.load(f)
    return {
        "train": set(raw.get("train", [])),
        "val": set(raw.get("validation", raw.get("val", raw.get("valid", [])))),
        "test": set(raw.get("test", [])),
    }


def _encode_sequence(seq):
    try:
        return torch.tensor([AA_TO_ID[aa] for aa in seq], dtype=torch.long)
    except KeyError:
        return None


def _stack_coords(entry):
    coords = entry.get("coords", {})
    if any(atom not in coords for atom in BACKBONE_ATOMS):
        return None, None

    try:
        stacked = np.stack(
            [np.asarray(coords[atom], dtype=np.float32) for atom in BACKBONE_ATOMS],
            axis=1,
        )
    except ValueError:
        return None, None
    if stacked.ndim != 3 or stacked.shape[1:] != (4, 3):
        return None, None

    valid_mask = np.isfinite(stacked).all(axis=(1, 2))
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(stacked), torch.from_numpy(valid_mask.astype(np.bool_))


def _process_entry(entry):
    name = entry.get("name")
    seq = entry.get("seq")
    if not name or not seq:
        return None
    if len(seq) > MAX_SEQ_LEN:
        return None

    seq_ids = _encode_sequence(seq)
    if seq_ids is None:
        return None

    coords, valid_mask = _stack_coords(entry)
    if coords is None or valid_mask is None:
        return None
    if coords.shape[0] != len(seq):
        return None
    if int(valid_mask.sum().item()) == 0:
        return None

    return {
        "name": name,
        "coords": coords.contiguous(),
        "targets": seq_ids.contiguous(),
        "valid_mask": valid_mask.contiguous(),
        "length": int(len(seq)),
    }


def build_processed_cache():
    """Process raw JSON into cached .pt files per split."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    expected = [os.path.join(PROCESSED_DIR, f"{s}.pt") for s in ("train", "val", "test")]
    if all(os.path.exists(p) for p in expected):
        print(f"Processed cache: already exists at {PROCESSED_DIR}")
        return

    print("Processing CATH 4.2 dataset...")
    split_names = _load_split_names()
    processed = {"train": [], "val": [], "test": []}
    kept = {"train": 0, "val": 0, "test": 0}
    discarded = 0

    start = time.time()
    with open(CHAIN_SET_PATH) as f:
        for line_idx, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                discarded += 1
                continue
            name = entry.get("name")
            split = None
            if name in split_names["train"]:
                split = "train"
            elif name in split_names["val"]:
                split = "val"
            elif name in split_names["test"]:
                split = "test"
            else:
                discarded += 1
                continue

            item = _process_entry(entry)
            if item is None:
                discarded += 1
                continue

            processed[split].append(item)
            kept[split] += 1

            if line_idx % 1000 == 0:
                elapsed = time.time() - start
                print(
                    f"  {line_idx} entries scanned in {elapsed:.1f}s "
                    f"(train={kept['train']}, val={kept['val']}, test={kept['test']})"
                )

    for split, items in processed.items():
        dest = os.path.join(PROCESSED_DIR, f"{split}.pt")
        tmp = dest + ".tmp"
        torch.save(items, tmp)
        os.rename(tmp, dest)
        print(f"  Saved {len(items)} {split} proteins")

    print(f"  Discarded {discarded} entries")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_processed_split(split):
    path = os.path.join(PROCESSED_DIR, f"{split}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed split: {path}. Run `uv run prepare.py` first.")
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass
class Batch:
    coords: torch.Tensor      # (B, L, 4, 3) backbone atom coordinates
    targets: torch.Tensor      # (B, L) amino acid indices, -1 for padding
    node_mask: torch.Tensor    # (B, L) valid residue mask


def _collate(items, device):
    batch_size = len(items)
    max_len = max(item["length"] for item in items)

    coords = torch.zeros(batch_size, max_len, 4, 3, dtype=torch.float32)
    targets = torch.full((batch_size, max_len), -1, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(items):
        length = item["length"]
        coords[i, :length] = item["coords"]
        valid = item["valid_mask"].bool()
        node_mask[i, :length] = valid
        targets[i, :length] = item["targets"]
        targets[i, :length][~valid] = -1

    return Batch(
        coords=coords.to(device),
        targets=targets.to(device),
        node_mask=node_mask.to(device),
    )


def make_dataloader(split, batch_max_residues, device, shuffle=True):
    """
    Infinite iterator over protein batches with length-aware batching.
    Groups proteins by similar length to minimize padding waste.
    Yields (Batch, epoch) tuples.
    """
    items = load_processed_split(split)
    lengths = [item["length"] for item in items]

    epoch = 1
    while True:
        if shuffle:
            noisy = np.array(lengths, dtype=np.float32) + np.random.uniform(0, 10, len(lengths))
            order = np.argsort(noisy)
        else:
            order = np.argsort(lengths)

        batch_indices = []
        current_max = 0

        for idx in order:
            length = lengths[int(idx)]
            proposed_max = max(current_max, length)
            if batch_indices and proposed_max * (len(batch_indices) + 1) > batch_max_residues:
                yield _collate([items[i] for i in batch_indices], device), epoch
                batch_indices = []
                current_max = 0
                proposed_max = length

            batch_indices.append(int(idx))
            current_max = proposed_max

        if batch_indices:
            yield _collate([items[i] for i in batch_indices], device), epoch
        epoch += 1

# ---------------------------------------------------------------------------
# Fixed validation (DO NOT CHANGE - this is the ground truth metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model, device, batch_max_residues, max_residues=EVAL_RESIDUES):
    """
    Fixed validation for protein inverse folding.

    Model contract: model(coords, node_mask) -> logits
        coords:    (B, L, 4, 3) float32 backbone coordinates
        node_mask: (B, L) bool valid-residue mask
        logits:    (B, L, 20) per-residue class logits

    Returns dict with:
        val_nll:        cross-entropy per valid residue (primary metric, lower is better)
        val_perplexity: exp(val_nll)
        seq_recovery:   fraction of correctly predicted amino acids
        total_residues: number of residues scored
    """
    model.eval()
    loader = make_dataloader("val", batch_max_residues=batch_max_residues, device=device, shuffle=False)

    total_nll = 0.0
    total_correct = 0
    total_residues = 0

    use_amp = str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    for batch, _epoch in loader:
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch.coords, batch.node_mask)
        else:
            logits = model(batch.coords, batch.node_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.targets.reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).reshape(batch.targets.shape)

        valid = batch.targets != -1
        correct = (logits.argmax(dim=-1) == batch.targets) & valid

        total_nll += float(loss[valid].sum().item())
        total_correct += int(correct.sum().item())
        total_residues += int(valid.sum().item())

        if total_residues >= max_residues:
            break

    if total_residues == 0:
        raise RuntimeError("Validation set produced zero valid residues.")

    val_nll = total_nll / total_residues
    return {
        "val_nll": val_nll,
        "val_perplexity": math.exp(min(val_nll, 700)),
        "seq_recovery": total_correct / total_residues,
        "total_residues": total_residues,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data()
    print()

    # Step 2: Process into cached splits
    build_processed_cache()
    print()

    print("Done! Ready to train.")
