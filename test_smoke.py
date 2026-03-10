"""
Smoke test for protein inverse folding pipeline. Runs entirely on CPU.

Exercises data pipeline, model forward pass, evaluator, and training loop
without requiring a GPU. If the CATH dataset is unavailable (no network or
server down), generates synthetic protein data so all logic tests still run.

Usage: uv run test_smoke.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from prepare import (
    AA_ALPHABET,
    BACKBONE_ATOMS,
    PROCESSED_DIR,
    _collate,
    _encode_sequence,
    _stack_coords,
    load_processed_split,
)


# ---------------------------------------------------------------------------
# Synthetic data generation (used when CATH download is unavailable)
# ---------------------------------------------------------------------------

def _generate_synthetic_protein(length):
    """Generate a single synthetic protein entry matching the processed format."""
    rng = np.random.default_rng()
    # Random backbone coordinates with realistic-ish spacing (~3.8A CA-CA)
    ca_coords = np.cumsum(rng.normal(0, 1.3, (length, 3)), axis=0)
    coords = np.zeros((length, 4, 3), dtype=np.float32)
    for i, offset in enumerate([(0, -0.5, 0), (0, 0, 0), (0, 0.5, 0), (0, 1.0, 0)]):
        coords[:, i, :] = ca_coords + np.array(offset)
    seq = "".join(rng.choice(list(AA_ALPHABET), size=length))
    targets = torch.tensor([ord(c) - ord("A") for c in seq], dtype=torch.long)
    # Remap to actual AA_TO_ID
    from prepare import AA_TO_ID
    targets = torch.tensor([AA_TO_ID[aa] for aa in seq], dtype=torch.long)
    return {
        "name": f"synthetic_{length}",
        "coords": torch.from_numpy(coords).contiguous(),
        "targets": targets.contiguous(),
        "valid_mask": torch.ones(length, dtype=torch.bool).contiguous(),
        "length": length,
    }


def _ensure_synthetic_data():
    """Create synthetic .pt files if real data is missing. Returns True if synthetic."""
    expected = [os.path.join(PROCESSED_DIR, f"{s}.pt") for s in ("train", "val", "test")]
    if all(os.path.exists(p) for p in expected):
        return False

    print("  Real CATH data not found, generating synthetic proteins...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    split_sizes = {"train": 200, "val": 50, "test": 50}
    for split, n in split_sizes.items():
        items = [
            _generate_synthetic_protein(int(rng.integers(30, 200)))
            for _ in range(n)
        ]
        dest = os.path.join(PROCESSED_DIR, f"{split}.pt")
        tmp = dest + ".tmp"
        torch.save(items, tmp)
        os.rename(tmp, dest)
        print(f"  Created {n} synthetic {split} proteins")

    return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_data_available():
    """Ensure processed data exists (real or synthetic)."""
    is_synthetic = _ensure_synthetic_data()

    counts = {}
    for split in ("train", "val", "test"):
        items = load_processed_split(split)
        assert len(items) > 0, f"{split} split is empty"
        item = items[0]
        assert item["coords"].shape[1:] == (4, 3), f"coords shape: {item['coords'].shape}"
        assert item["targets"].ndim == 1
        assert item["coords"].shape[0] == item["targets"].shape[0]
        assert item["valid_mask"].shape[0] == item["targets"].shape[0]
        counts[split] = len(items)

    label = "synthetic" if is_synthetic else "real CATH"
    return f"{label}: {counts}"


def test_encode_sequence():
    """Verify amino acid encoding and rejection of non-standard residues."""
    from prepare import AA_TO_ID

    encoded = _encode_sequence("ACDEF")
    assert encoded is not None
    assert len(encoded) == 5
    assert encoded[0].item() == AA_TO_ID["A"]
    assert encoded[4].item() == AA_TO_ID["F"]

    # Full alphabet round-trip
    full = _encode_sequence(AA_ALPHABET)
    assert full is not None
    assert len(full) == 20
    for i in range(20):
        assert full[i].item() == i

    # Non-standard amino acid should return None
    assert _encode_sequence("ACXEF") is None

    # Empty string
    encoded_empty = _encode_sequence("")
    assert encoded_empty is not None and len(encoded_empty) == 0


def test_stack_coords():
    """Verify coordinate stacking, NaN handling, and missing-atom rejection."""
    L = 10
    entry = {
        "coords": {
            "N": np.random.randn(L, 3).tolist(),
            "CA": np.random.randn(L, 3).tolist(),
            "C": np.random.randn(L, 3).tolist(),
            "O": np.random.randn(L, 3).tolist(),
        }
    }
    coords, valid = _stack_coords(entry)
    assert coords is not None
    assert coords.shape == (L, 4, 3)
    assert valid.shape == (L,)
    assert valid.all()

    # Missing atom should return None
    bad_entry = {"coords": {"N": [[0, 0, 0]], "CA": [[0, 0, 0]]}}
    c, v = _stack_coords(bad_entry)
    assert c is None

    # NaN coordinate should mark residue as invalid
    nan_entry = {
        "coords": {
            "N": [[float("nan"), 0, 0], [1, 2, 3]],
            "CA": [[0, 0, 0], [1, 2, 3]],
            "C": [[0, 0, 0], [1, 2, 3]],
            "O": [[0, 0, 0], [1, 2, 3]],
        }
    }
    coords, valid = _stack_coords(nan_entry)
    assert coords is not None
    assert not valid[0].item()  # first residue has NaN -> invalid
    assert valid[1].item()      # second residue is fine

    # Mismatched lengths should return None
    bad_lengths = {
        "coords": {
            "N": [[0, 0, 0], [1, 1, 1]],
            "CA": [[0, 0, 0]],
            "C": [[0, 0, 0], [1, 1, 1]],
            "O": [[0, 0, 0], [1, 1, 1]],
        }
    }
    c, v = _stack_coords(bad_lengths)
    assert c is None


def test_collate():
    """Verify batch collation pads correctly and sets ignore-index targets."""
    items = [
        {
            "coords": torch.randn(5, 4, 3),
            "targets": torch.randint(0, 20, (5,)),
            "valid_mask": torch.ones(5, dtype=torch.bool),
            "length": 5,
        },
        {
            "coords": torch.randn(8, 4, 3),
            "targets": torch.randint(0, 20, (8,)),
            "valid_mask": torch.ones(8, dtype=torch.bool),
            "length": 8,
        },
    ]
    batch = _collate(items, "cpu")
    assert batch.coords.shape == (2, 8, 4, 3)
    assert batch.targets.shape == (2, 8)
    assert batch.node_mask.shape == (2, 8)
    # First item (length 5) should be padded with -1 targets after position 5
    assert batch.targets[0, 5:].eq(-1).all()
    assert batch.node_mask[0, :5].all()
    assert not batch.node_mask[0, 5:].any()

    # Single item batch
    single = _collate([items[0]], "cpu")
    assert single.coords.shape == (1, 5, 4, 3)


def test_residue_features():
    """Verify geometric feature extraction shapes and masking."""
    from train import _residue_features

    B, L = 2, 10
    coords = torch.randn(B, L, 4, 3)
    mask = torch.ones(B, L, dtype=torch.bool)

    features = _residue_features(coords, mask)
    assert features.shape == (B, L, 22), f"Expected (2, 10, 22), got {features.shape}"

    # Masked positions should produce zero features
    mask[0, 3] = False
    features = _residue_features(coords, mask)
    assert features[0, 3].abs().sum().item() == 0.0

    # Neighbors of masked positions should have zeroed CA-CA diffs
    # The CA-CA diff components are the last 6 features (indices 16-21)
    ca_diff_prev_4 = features[0, 4, 16:19]  # ca_diff_prev for position 4
    ca_diff_next_2 = features[0, 2, 19:22]  # ca_diff_next for position 2
    assert ca_diff_prev_4.abs().sum().item() == 0.0, "Neighbor of masked should have zeroed ca_diff_prev"
    assert ca_diff_next_2.abs().sum().item() == 0.0, "Neighbor of masked should have zeroed ca_diff_next"


def test_model_forward():
    """Verify model instantiation and forward pass output shapes."""
    from train import ProteinTransformer, ModelConfig

    config = ModelConfig(d_model=64, nhead=4, num_layers=2, mlp_ratio=2)
    model = ProteinTransformer(config)

    B, L = 2, 15
    coords = torch.randn(B, L, 4, 3)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 10:] = False

    logits = model(coords, mask)
    assert logits.shape == (B, L, 20), f"Expected (2, 15, 20), got {logits.shape}"

    # Loss should be computable and backprop should work
    targets = torch.randint(0, 20, (B, L))
    targets[0, 10:] = -1
    loss = F.cross_entropy(logits.reshape(-1, 20), targets.reshape(-1), ignore_index=-1)
    assert loss.item() > 0
    loss.backward()

    # Verify all parameters got gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    num_params = sum(p.numel() for p in model.parameters())
    return f"{num_params:,} params"


def test_lr_schedule():
    """Verify LR schedule has expected warmup/plateau/cooldown shape."""
    from train import get_lr_multiplier, FINAL_LR_FRAC

    # Warmup: starts near 0
    assert get_lr_multiplier(0.0) < 0.01

    # Mid-warmup: should be increasing
    assert get_lr_multiplier(0.025) > get_lr_multiplier(0.0)

    # Plateau: should be 1.0
    assert get_lr_multiplier(0.5) == 1.0

    # Cooldown: should be decreasing
    assert get_lr_multiplier(0.9) < 1.0
    assert get_lr_multiplier(0.95) < get_lr_multiplier(0.9)

    # End: approaches FINAL_LR_FRAC
    assert abs(get_lr_multiplier(1.0) - FINAL_LR_FRAC) < 0.01

    # Monotonicity check across full range
    values = [get_lr_multiplier(p / 100) for p in range(101)]
    # Warmup phase should be non-decreasing
    warmup_end = 5  # WARMUP_RATIO = 0.05
    for i in range(1, warmup_end + 1):
        assert values[i] >= values[i - 1], f"Warmup not monotonic at {i}%"


def test_evaluator():
    """Run validation with a small residue cap on CPU. This is the key test."""
    from prepare import run_validation
    from train import ProteinTransformer, ModelConfig

    config = ModelConfig(d_model=64, nhead=4, num_layers=2, mlp_ratio=2)
    model = ProteinTransformer(config)

    metrics = run_validation(
        model, "cpu", batch_max_residues=2048, max_residues=1000
    )

    assert "val_nll" in metrics
    assert "val_perplexity" in metrics
    assert "seq_recovery" in metrics
    assert "total_residues" in metrics

    assert metrics["val_nll"] > 0, f"val_nll should be positive, got {metrics['val_nll']}"
    assert metrics["val_perplexity"] > 1.0, "Perplexity should be > 1"
    assert 0 <= metrics["seq_recovery"] <= 1.0
    assert metrics["total_residues"] >= 1000

    # Random model should have ~uniform predictions -> perplexity near 20
    assert metrics["val_perplexity"] > 5, (
        f"Random model perplexity suspiciously low: {metrics['val_perplexity']}"
    )

    return {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()}


def test_training_steps():
    """Run 3 training steps on CPU to verify the full loop mechanics."""
    from train import ProteinTransformer, ModelConfig, LEARNING_RATE
    from prepare import make_dataloader

    config = ModelConfig(d_model=64, nhead=4, num_layers=2, mlp_ratio=2)
    model = ProteinTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    loader = make_dataloader("train", batch_max_residues=2048, device="cpu", shuffle=True)

    losses = []
    for _step in range(3):
        model.train()
        batch, _epoch = next(loader)
        logits = model(batch.coords, batch.node_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch.targets.reshape(-1),
            ignore_index=-1,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())

    assert all(l > 0 for l in losses), f"All losses should be positive: {losses}"
    assert all(l < 100 for l in losses), f"Losses exploded: {losses}"
    return [f"{l:.4f}" for l in losses]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        ("Data available", test_data_available),
        ("Sequence encoding", test_encode_sequence),
        ("Coordinate stacking", test_stack_coords),
        ("Batch collation", test_collate),
        ("Geometric features", test_residue_features),
        ("Model forward pass", test_model_forward),
        ("LR schedule", test_lr_schedule),
        ("Evaluator (1k residues)", test_evaluator),
        ("Training loop (3 steps)", test_training_steps),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        t0 = time.time()
        try:
            result = fn()
            dt = time.time() - t0
            print(f"  PASS  {name} ({dt:.1f}s)")
            if result is not None:
                if isinstance(result, dict):
                    for k, v in result.items():
                        print(f"        {k}: {v}")
                elif isinstance(result, list):
                    print(f"        -> {result}")
                else:
                    print(f"        -> {result}")
            passed += 1
        except Exception as e:
            dt = time.time() - t0
            print(f"  FAIL  {name} ({dt:.1f}s)")
            print(f"        {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    return failed == 0


if __name__ == "__main__":
    print("Smoke test: protein inverse folding pipeline (CPU)")
    print("=" * 55)
    print()

    success = run_all()
    sys.exit(0 if success else 1)
