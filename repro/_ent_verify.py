"""Live entropy verification for paper claims (L3 v2 vs L2mH control)."""
import sys
sys.path.insert(0, "src")
import numpy as np
from jepa_id.worlds import L3World, L2World

def ent(col):
    _, c = np.unique(col, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

w3 = L3World(seed=0, obs_per_pos=1)
d = w3.generate(100000)
z = d["z"]
perdim = [ent(z[:, j]) for j in range(3)]
rows, c = np.unique(z, axis=0, return_counts=True)
p = c / c.sum()
joint = float(-(p * np.log2(p)).sum())
print("L3 v2 per-dim H:", [round(x, 3) for x in perdim], "joint H:", round(joint, 3), "states:", len(rows))

w2 = L2World(latent_dim=3, S=(2, 4, 4), seed=0)
dz = w2.generate(100000)
perdim2 = [ent(dz["z"][:, j]) for j in range(3)]
rows2, c2 = np.unique(dz["z"], axis=0, return_counts=True)
p2 = c2 / c2.sum()
joint2 = float(-(p2 * np.log2(p2)).sum())
print("L2mH(2,4,4) per-dim H:", [round(x, 3) for x in perdim2], "joint H:", round(joint2, 3), "states:", len(rows2))

w2b = L2World(latent_dim=3, S=2, seed=0)
dzb = w2b.generate(100000)
rowsb, cb = np.unique(dzb["z"], axis=0, return_counts=True)
pb = cb / cb.sum()
print("L2mS2 per-dim H:", [round(ent(dzb['z'][:, j]), 3) for j in range(3)],
      "joint H:", round(float(-(pb * np.log2(pb)).sum()), 3), "states:", len(rowsb))
