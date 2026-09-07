"""Staged factorial grid runner for the identifiability ladder.

Usage:
    PYTHONPATH=src uv run python repro/run_sweep.py --stage pilot --device cuda
    PYTHONPATH=src uv run python repro/run_sweep.py --stage 1 --device cuda
    PYTHONPATH=src uv run python repro/run_sweep.py --stage 2 --device cuda [--limit 30]
    PYTHONPATH=src uv run python repro/run_sweep.py --stage 3 --device cuda

Design follows pre_registration_2026-09-06.md sections 7-8 (factorial
structure, staged launch). Each cell is one independent training run (seed =>
independent optimizer trajectory); rows are appended to a JSONL file, keyed by
cell_id for resumability (re-running skips completed cells).

Amendment A2 (2026-09-07, Gate 0b evidence): the pre-reg Stage-1 plan said
"linear g"; Gate 0b showed linear mixing is degenerate for L0 (R^2 ~= 0.99 flat
across all alpha — nothing to discriminate). Stages therefore run nonlinear g
by default (the discriminating regime); the linear arm returns only in the full
factorial (Stage 4-equivalent, deferred).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from jepa_id.train import TrainConfig, train, evaluate
from jepa_id.worlds import alpha_grid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CKPT = ROOT / "results" / "ckpts"

MODELS = ["jepa", "recon", "contrastive"]
SEEDS_FULL = [0, 1, 2, 3, 4]
SEEDS_PILOT = [0, 1, 2]
DIM_FULL = [8, 32]
DIM_PILOT = [8]
G_FULL = ["nonlinear"]   # Amendment A2 — see docstring
K_GRID = [2, 4, 8, 16]
S_GRID = [3, 5, 10]


def cell_id(c: dict) -> str:
    return "-".join(f"{k}{v}" for k, v in sorted(c.items()))


def stage_cells(stage: str, limit: int | None = None) -> list[dict]:
    cells = []
    if stage == "pilot":
        # 3 worlds x 3 models x 3 seeds — runner validation + cost calibration
        for world, dial in [("L0", {"alpha": 2.0}), ("L1", {"K": 8}), ("L2", {"S": 5})]:
            for model in MODELS:
                for seed in SEEDS_PILOT:
                    for dim in DIM_PILOT:
                        cells.append({"stage": stage, "world": world, **dial,
                                      "g": "nonlinear", "model": model,
                                      "seed": seed, "dim": dim})
    elif stage == "1":
        # Stage 1 (fast-pass): L0 all alpha + L1 all K, 3 seeds, nonlinear g.
        for world in ["L0"]:
            for a in alpha_grid():
                for model in MODELS:
                    for seed in SEEDS_PILOT:
                        cells.append({"stage": stage, "world": world, "alpha": float(a),
                                      "g": "nonlinear", "model": model,
                                      "seed": seed, "dim": 8})
        for K in K_GRID:
            for model in MODELS:
                for seed in SEEDS_PILOT:
                    cells.append({"stage": stage, "world": "L1", "K": K,
                                  "g": "nonlinear", "model": model,
                                  "seed": seed, "dim": 8})
    elif stage == "1b":
        # Stage 1b (A3 re-run, official gate): L0 all alpha, spiral 2D, 3 seeds.
        # jepa: sigreg_lambda=10 (SIGReg must actually Gaussian-force); contrastive:
        # temperature=0.3 (InfoNCE peak at alpha=2). Recon excluded (autoencoder,
        # not a Fig-4b objective — see Amendment A3).
        for a in alpha_grid():
            for model in ["jepa", "contrastive"]:
                for seed in SEEDS_PILOT:
                    cells.append({"stage": stage, "world": "L0", "alpha": float(a),
                                  "g": "spiral", "model": model, "seed": seed,
                                  "dim": 2, "amp": 1.0,
                                  "sigreg_lambda": 10.0 if model == "jepa" else 1.0,
                                  "temperature": 0.3 if model == "contrastive" else 0.1})
        # L1 discrete bridge re-run with spiral 2D for consistency of the report.
        for K in K_GRID:
            for model in ["jepa", "contrastive"]:
                for seed in SEEDS_PILOT:
                    cells.append({"stage": stage, "world": "L1", "K": K,
                                  "g": "spiral", "model": model, "seed": seed,
                                  "dim": 2, "amp": 1.0,
                                  "sigreg_lambda": 10.0 if model == "jepa" else 1.0,
                                  "temperature": 0.3 if model == "contrastive" else 0.1})
    elif stage == "2":
        # Stage 2 (discrete core): L1 (full K x dim x seed) + L2 (S dial)
        # Mixing: spiral (A3) — the quadratic map is uninvertible for small MLPs.
        for K in K_GRID:
            for model in MODELS:
                for seed in SEEDS_FULL:
                    for dim in DIM_FULL:
                        cells.append({"stage": stage, "world": "L1", "K": K,
                                      "g": "spiral", "model": model, "amp": 1.0,
                                      "seed": seed, "dim": dim})
        for S in S_GRID:
            for model in MODELS:
                for seed in SEEDS_FULL:
                    for dim in DIM_FULL:
                        cells.append({"stage": stage, "world": "L2", "S": S,
                                      "g": "spiral", "model": model, "amp": 1.0,
                                      "seed": seed, "dim": dim})
        # L3 rule-governed grammar (fixed structure)
        for model in MODELS:
            for seed in SEEDS_FULL:
                cells.append({"stage": stage, "world": "L3", "g": "spiral",
                              "model": model, "seed": seed, "dim": 6})
    elif stage == "3":
        # Stage 3: L4 Turkish morphology (fixed structure)
        for model in MODELS:
            for seed in SEEDS_FULL:
                cells.append({"stage": stage, "world": "L4", "g": "nonlinear",
                              "model": model, "seed": seed, "dim": 3})
    else:
        raise ValueError(f"unknown stage: {stage}")
    if limit:
        cells = cells[:limit]
    return cells


def _jsonable(v):
    """Recursively convert numpy scalars/arrays to JSON-native types."""
    if isinstance(v, dict):
        return {kk: _jsonable(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(vv) for vv in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def run_cell(c: dict, device: str) -> dict:
    t0 = time.time()
    cid = cell_id(c)
    kw = {"world": c["world"], "model": c["model"], "seed": c["seed"],
          "latent_dim": c["dim"], "device": device,
          "emb_dim": c.get("emb", c["dim"]),   # faithful A3: emb == latent dim
          "log_path": str(CKPT / f"train_{cid}.jsonl"),
          "ckpt_path": str(CKPT / f"ckpt_{cid}.pt")}
    for k in ("alpha", "K", "S"):
        if k in c:
            kw[k] = c[k]
    if "g" in c:
        kw["mixing"] = c["g"]
    if "sigreg_lambda" in c:
        kw["sigreg_lambda"] = c["sigreg_lambda"]
    if "temperature" in c:
        kw["temperature"] = c["temperature"]
    if "amp" in c:
        kw["amp"] = c["amp"]
    cfg = TrainConfig(**kw)
    train(cfg)
    ev = _jsonable(evaluate(str(CKPT / f"ckpt_{cid}.pt")))
    row = {**c, "metrics": ev, "time_s": round(time.time() - t0, 1)}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS / f"sweep_stage{args.stage}.jsonl"

    cells = stage_cells(args.stage, args.limit)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["cell_id"])
            except json.JSONDecodeError:
                continue

    n_run = 0
    for c in cells:
        cid = cell_id(c)
        if cid in done:
            continue
        try:
            row = run_cell(c, args.device)
        except Exception as e:  # keep the batch alive; log and continue
            row = {**c, "error": f"{type(e).__name__}: {e}", "time_s": None}
            print(f"ERROR {cid}: {type(e).__name__}: {e}", flush=True)
        row["cell_id"] = cid
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        n_run += 1
        print(f"done {cid} ({n_run} this run)", flush=True)
    print(f"stage {args.stage}: {n_run} new cells -> {out_path}")


if __name__ == "__main__":
    main()
