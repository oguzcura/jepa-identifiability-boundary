"""Training loop for the identifiability ladder.

Trains a single model/objective on a single world, deterministic-seeded,
GPU-aware, with checkpointing. Reads data from the world generators in
`worlds.py` and writes per-step/epoch metrics to a JSONL log.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from jepa_id.models import build_model
from jepa_id.worlds import WORLD_REGISTRY


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


@dataclass
class TrainConfig:
    world: str = "L0"
    model: str = "jepa"          # jepa | recon | contrastive
    emb_dim: int = 64
    hidden: int = 256
    depth: int = 2
    alpha: float = 2.0           # L0 gennorm shape (ignored for L1-L4)
    K: int = 8                   # L1 bins
    S: int = 5                   # L2 categories
    batch_size: int = 256
    steps: int = 2000
    lr: float = 1e-3
    tau: float = 0.99
    seed: int = 0
    device: str = "cpu"
    log_path: str = "results/{world}_{model}_alpha{alpha}_seed{seed}.jsonl"
    ckpt_path: str = "results/{world}_{model}_alpha{alpha}_seed{seed}.pt"
    log_every: int = 100


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device

    # Build world
    wcls = WORLD_REGISTRY[cfg.world]
    if cfg.world == "L0":
        world = wcls(latent_dim=4, alpha=cfg.alpha, seed=cfg.seed)
        obs_dim = 4
    elif cfg.world == "L1":
        world = wcls(latent_dim=4, K=cfg.K, seed=cfg.seed)
        obs_dim = 4
    elif cfg.world == "L2":
        world = wcls(latent_dim=4, S=cfg.S, seed=cfg.seed)
        obs_dim = 4
    else:  # L3, L4
        world = wcls(seed=cfg.seed)
        obs_dim = world.generate(1)["x"].shape[1]

    model = build_model(cfg.model, obs_dim=obs_dim, emb_dim=cfg.emb_dim,
                        hidden=cfg.hidden, depth=cfg.depth,
                        tau=cfg.tau, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    log_path = Path(cfg.log_path.format(**cfg.__dict__))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    model.train()
    for step in range(cfg.steps):
        d = world.generate(cfg.batch_size)
        x = _to_tensor(d["x"], device)
        xp = _to_tensor(d["x_prime"], device)
        loss, h, *_ = model(x, xp)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            row = {"step": step, "loss": float(loss.detach().cpu()), "elapsed": time.time() - t0}
            rows.append(row)
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    ckpt_path = Path(cfg.ckpt_path.format(**cfg.__dict__))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

    return {"final_loss": rows[-1]["loss"], "steps": cfg.steps, "ckpt": str(ckpt_path)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="L0")
    p.add_argument("--model", default="jepa")
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--emb-dim", type=int, default=64)
    args = p.parse_args()
    cfg = TrainConfig(world=args.world, model=args.model, alpha=args.alpha,
                      steps=args.steps, batch_size=args.batch_size,
                      seed=args.seed, device=args.device, emb_dim=args.emb_dim)
    print(json.dumps(train(cfg)))


if __name__ == "__main__":
    main()
