"""
Protein inverse folding training script. Single-GPU, single-file.
Given backbone coordinates (N, CA, C, O), predict the amino acid sequence.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from prepare import NUM_CLASSES, MAX_SEQ_LEN, TIME_BUDGET, make_dataloader, run_validation

# ---------------------------------------------------------------------------
# Protein Inverse Folding Model
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    max_positions: int = 1024
    num_classes: int = 20


def _residue_features(coords, node_mask):
    """
    Extract local geometric features from backbone coordinates.

    Input:  coords (B, L, 4, 3) -- N, CA, C, O positions
            node_mask (B, L) -- valid residue mask
    Output: features (B, L, 22) -- per-residue geometric features
    """
    B, L, _, _ = coords.shape

    # Center on CA
    ca = coords[:, :, 1, :]  # (B, L, 3)
    centered = coords - ca.unsqueeze(2)  # (B, L, 4, 3)
    flat_coords = centered.reshape(B, L, 12)  # 12D

    # Radial distances from CA to each atom
    radii = centered.norm(dim=-1)  # (B, L, 4)

    # CA-CA differences (prev and next), masked by BOTH current AND neighbor validity
    ca_diff_prev = torch.zeros_like(ca)
    ca_diff_next = torch.zeros_like(ca)
    ca_diff_prev[:, 1:] = ca[:, 1:] - ca[:, :-1]
    ca_diff_next[:, :-1] = ca[:, 1:] - ca[:, :-1]

    # Mask: require both the current residue and its neighbor to be valid
    valid_f = node_mask.float()
    mask_prev = torch.zeros_like(valid_f)
    mask_next = torch.zeros_like(valid_f)
    mask_prev[:, 1:] = valid_f[:, 1:] * valid_f[:, :-1]
    mask_next[:, :-1] = valid_f[:, :-1] * valid_f[:, 1:]
    ca_diff_prev = ca_diff_prev * mask_prev.unsqueeze(-1)
    ca_diff_next = ca_diff_next * mask_next.unsqueeze(-1)

    # Concatenate: 12 + 4 + 3 + 3 = 22 features
    features = torch.cat([flat_coords, radii, ca_diff_prev, ca_diff_next], dim=-1)
    features = features * node_mask.unsqueeze(-1).float()
    return features


class ProteinTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        # Input projection: 22D geometric features -> d_model
        self.input_proj = nn.Linear(22, d)

        # Positional embeddings
        self.pos_emb = nn.Embedding(config.max_positions, d)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.nhead,
            dim_feedforward=d * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(d)

        # Output head
        self.head = nn.Linear(d, config.num_classes)

    def forward(self, coords, node_mask):
        B, L = coords.shape[:2]

        # Extract geometric features
        features = _residue_features(coords, node_mask)

        # Project and add positional embeddings
        x = self.input_proj(features)
        positions = torch.arange(L, device=coords.device)
        x = x + self.pos_emb(positions)

        # Create attention mask (mask out padding)
        src_key_padding_mask = ~node_mask

        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Predict
        logits = self.head(x)
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
DEPTH = 6               # number of transformer layers
D_MODEL = 256           # model dimension
NHEAD = 8               # attention heads
MLP_RATIO = 4           # feedforward expansion
DROPOUT = 0.1           # dropout rate

# Optimization
LEARNING_RATE = 3e-4    # AdamW learning rate
WEIGHT_DECAY = 1e-2     # L2 regularization
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.01   # final LR as fraction of initial

# Batching
BATCH_MAX_RESIDUES = 4096  # max residues per batch (length-aware)

# ---------------------------------------------------------------------------
# LR schedule (pure function, importable for testing)
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Main: setup, training loop, final scoring
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)

    config = ModelConfig(
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=DEPTH,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        max_positions=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES,
    )
    print(f"Model config: {asdict(config)}")

    model = ProteinTransformer(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if device.type == "cuda" and not os.environ.get("TORCHDYNAMO_DISABLE"):
        try:
            model = torch.compile(model, dynamic=False)
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation")

    train_loader = make_dataloader("train", batch_max_residues=BATCH_MAX_RESIDUES, device=str(device), shuffle=True)
    batch, epoch = next(train_loader)  # prefetch first batch

    # TensorBoard setup
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_d{D_MODEL}_L{DEPTH}_h{NHEAD}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("config", str(asdict(config)))
    print(f"TensorBoard: {log_dir}")

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Device: {device}")

    t_start_training = time.time()
    smooth_train_loss = 0
    total_training_time = 0
    step = 0

    while True:
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with autocast_ctx:
            logits = model(batch.coords, batch.node_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.targets.reshape(-1),
                ignore_index=-1,
            )

        train_loss = loss.detach()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # LR schedule
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * lrm

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch, epoch = next(train_loader)

        train_loss_f = train_loss.item()

        # Fast fail: abort if loss is exploding
        if train_loss_f > 100:
            print("FAIL")
            exit(1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # TensorBoard logging
        writer.add_scalar("train/loss_raw", train_loss_f, step)
        writer.add_scalar("train/loss_smooth", debiased_smooth_loss, step)
        writer.add_scalar("train/lr_multiplier", lrm, step)
        writer.add_scalar("train/step_ms", dt * 1000, step)
        writer.add_scalar("train/epoch", epoch, step)
        writer.add_scalar("train/progress", progress, step)
        if device.type == "cuda":
            writer.add_scalar("system/vram_mb", torch.cuda.memory_allocated() / 1024 / 1024, step)

        # GC management (Python's GC causes stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        # Time's up -- but only stop after warmup steps so we don't count compilation
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()  # newline after \r training log

    # Final scoring
    model.eval()
    with autocast_ctx:
        metrics = run_validation(model, str(device), batch_max_residues=BATCH_MAX_RESIDUES)

    # Final summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

    # TensorBoard: final validation metrics
    writer.add_scalar("val/nll", metrics["val_nll"], step)
    writer.add_scalar("val/perplexity", metrics["val_perplexity"], step)
    writer.add_scalar("val/seq_recovery", metrics["seq_recovery"], step)

    # TensorBoard: hyperparameter/metric table for cross-run comparison
    writer.add_hparams(
        hparam_dict={
            "d_model": D_MODEL,
            "depth": DEPTH,
            "nhead": NHEAD,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_max_residues": BATCH_MAX_RESIDUES,
            "num_params_M": num_params / 1e6,
        },
        metric_dict={
            "hparam/val_nll": metrics["val_nll"],
            "hparam/val_perplexity": metrics["val_perplexity"],
            "hparam/seq_recovery": metrics["seq_recovery"],
            "hparam/num_steps": step,
        },
        run_name=".",
    )
    writer.close()

    print("---")
    print(f"val_nll:          {metrics['val_nll']:.6f}")
    print(f"val_perplexity:   {metrics['val_perplexity']:.6f}")
    print(f"seq_recovery:     {metrics['seq_recovery']:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")

    # Append to results.tsv for cross-experiment tracking
    results_path = os.path.join(os.path.dirname(__file__), "results.tsv")
    header_needed = not os.path.exists(results_path)
    with open(results_path, "a") as f:
        if header_needed:
            f.write("run\tval_nll\tval_perplexity\tseq_recovery\t"
                    "steps\ttraining_s\td_model\tdepth\tnhead\tlr\t"
                    "dropout\tbatch_residues\tparams_M\tpeak_vram_mb\n")
        f.write(f"{run_name}\t{metrics['val_nll']:.6f}\t{metrics['val_perplexity']:.6f}\t"
                f"{metrics['seq_recovery']:.6f}\t{step}\t{total_training_time:.1f}\t"
                f"{D_MODEL}\t{DEPTH}\t{NHEAD}\t{LEARNING_RATE}\t"
                f"{DROPOUT}\t{BATCH_MAX_RESIDUES}\t{num_params / 1e6:.1f}\t"
                f"{peak_vram_mb:.1f}\n")
    print(f"Results appended to {results_path}")
