"""Gate 0b: reproduce the Fig 4b trend — JEPA linear recovery across gennorm alpha.

Runs the alpha sweep (2^-3 ... 2^5), trains JEPA on each, evaluates linear R^2
of z from h, verifies the peak is at alpha=2 (Gaussian), per the theorem
(paper Sec 6.2 / App H.7).
"""

import json

from jepa_id.train import train, TrainConfig, evaluate
from jepa_id.worlds import alpha_grid


def run(device="cuda", steps=1200, seed=0, batch_size=512):
    results = []
    for alpha in alpha_grid():
        cfg = TrainConfig(world="L0", model="jepa", alpha=float(alpha),
                          steps=steps, batch_size=batch_size, emb_dim=64,
                          seed=seed, device=device)
        tr = train(cfg)
        ev = evaluate(tr["ckpt"], seed=1)
        results.append({
            "alpha": float(alpha),
            "final_loss": tr["final_loss"],
            "r2_mean": ev["ridge"]["mean"],
            "r2_pooled": ev["ridge"]["pooled"],
            "cca": ev["cca"]["mean"],
            "mlp_control": ev["mlp_control"],
            "pred_loss": ev["pred_loss"],
            "eff_rank": ev["collapse"]["effective_rank"],
        })
        print(json.dumps({"alpha": float(alpha), "r2": round(ev["ridge"]["mean"], 4),
                          "loss": round(tr["final_loss"], 4)}), flush=True)
    peak = max(results, key=lambda r: r["r2_mean"])
    print(f"\nPEAK at alpha={peak['alpha']} (r2={peak['r2_mean']:.4f}) — expect 2.0")
    with open("results/gate0b_alpha_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    import sys
    run(device=sys.argv[1] if len(sys.argv) > 1 else "cuda")