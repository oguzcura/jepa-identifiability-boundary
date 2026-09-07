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
from jepa_id.readout import evaluate_identifiability
from jepa_id.worlds import WORLD_REGISTRY


def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)


def _standardize(x: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Z-score observations (data-independent scale fix; mixing can create
    columns with wildly different scales, which destabilizes MLP training)."""
    return (x - mu) / (sd + 1e-6)


def _fit_scale(world, n: int = 4000, seed: int = 123) -> tuple:
    """Fit per-dim mean/std of observations from a deterministic sample."""
    d = world.generate(n)
    x = np.asarray(d["x"], dtype=np.float32)
    return torch.tensor(x.mean(0), dtype=torch.float32), torch.tensor(x.std(0), dtype=torch.float32)


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
    mixing: str = "nonlinear"    # linear | nonlinear (spiral mixing; theorem uses nonlinear)
    sigreg_lambda: float = 1.0   # SIGReg weight (JEPA only; ignored by recon/contrastive)
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
        world = wcls(latent_dim=4, alpha=cfg.alpha, g=cfg.mixing, seed=cfg.seed)
        obs_dim = 4
    elif cfg.world == "L1":
        world = wcls(latent_dim=4, K=cfg.K, g=cfg.mixing, seed=cfg.seed)
        obs_dim = 4
    elif cfg.world == "L2":
        world = wcls(latent_dim=4, S=cfg.S, g=cfg.mixing, seed=cfg.seed)
        obs_dim = 4
    else:  # L3, L4
        world = wcls(seed=cfg.seed)
        obs_dim = world.generate(1)["x"].shape[1]

    sigreg_lambda = getattr(cfg, "sigreg_lambda", 1.0)
    model = build_model(cfg.model, obs_dim=obs_dim, emb_dim=cfg.emb_dim,
                        hidden=cfg.hidden, depth=cfg.depth,
                        tau=cfg.tau, sigreg_lambda=sigreg_lambda,
                        device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # data-independent observation standardization (fixes scale instability)
    xmu, xsd = _fit_scale(world)
    xmu, xsd = xmu.to(device), xsd.to(device)

    log_path = Path(cfg.log_path.format(**cfg.__dict__))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    model.train()
    for step in range(cfg.steps):
        d = world.generate(cfg.batch_size)
        x = _standardize(_to_tensor(d["x"], device), xmu, xsd)
        xp = _standardize(_to_tensor(d["x_prime"], device), xmu, xsd)
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


def evaluate(cfg_ckpt: str, n_eval: int = 2000, seed: int = 1) -> dict:
    """Load a trained checkpoint, embed held-out world data, run readouts.

    Returns the full identifiability battery plus prediction loss (the
    'decoupling' pair: prediction quality vs. identifiability).
    """
    import torch as T
    ckpt = T.load(cfg_ckpt, map_location="cpu", weights_only=False)
    cfgd = ckpt["cfg"]
    world = cfgd["world"]
    wcls = WORLD_REGISTRY[world]
    mixing = cfgd.get("mixing", "nonlinear")
    if world == "L0":
        world_obj = wcls(latent_dim=4, alpha=cfgd["alpha"], g=mixing, seed=seed)
        obs_dim = 4
    elif world == "L1":
        world_obj = wcls(latent_dim=4, K=cfgd["K"], g=mixing, seed=seed)
        obs_dim = 4
    elif world == "L2":
        world_obj = wcls(latent_dim=4, S=cfgd["S"], g=mixing, seed=seed)
        obs_dim = 4
    else:
        world_obj = wcls(seed=seed)
        obs_dim = world_obj.generate(1)["x"].shape[1]

    from jepa_id.models import build_model
    import inspect
    sig = inspect.signature(build_model)
    m = build_model(cfgd["model"], obs_dim=obs_dim, emb_dim=cfgd["emb_dim"],
                    hidden=cfgd.get("hidden", 256), depth=cfgd.get("depth", 2))
    m.load_state_dict(ckpt["state_dict"])
    m.eval()

    d = world_obj.generate(n_eval)
    xmu, xsd = _fit_scale(world_obj, n=4000)
    x = _standardize(T.tensor(d["x"], dtype=T.float32), xmu, xsd)
    with T.no_grad():
        h = m.encoder(x).numpy() if cfgd["model"] != "jepa" else m.context(x).numpy()
    z = d["z"]
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    out = evaluate_identifiability(h, z, seed=seed)
    # prediction loss (decoupling plot): mse on objective on held-out data
    with T.no_grad():
        xp = _standardize(T.tensor(d["x_prime"], dtype=T.float32), xmu, xsd)
        loss, *_ = m(x, xp, momentum=False)
    out["pred_loss"] = float(loss.item())
    out["world"] = world
    out["model"] = cfgd["model"]
    return out


if __name__ == "__main__":
    import glob
    ckpts = sorted(glob.glob("results/*.pt"))
    print(f"{len(ckpts)} checkpoints found")
    for ck in ckpts[:3]:
        print(ck, "->", {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in evaluate(ck).items() if k not in ("per_dim", "canonical_corrs")})
