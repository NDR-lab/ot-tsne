import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from toy_utils import make_circle_data, compute_ot_plan

# --- params ---
n0 = 1000
n1 = 5000
seed = 0
eps_times = 1
n_iter = 10000
out_dir = os.path.join(os.path.dirname(__file__), "..", "data")

# --- data ---
print(f"generating circle data: n0={n0}, n1={n1}, seed={seed}")
x0, x1 = make_circle_data(n0=n0, n1=n1, seed=seed)

# --- ot plan ---
print(f"computing OT plan (eps_times={eps_times}, n_iter={n_iter}) ...")
P = compute_ot_plan(x0, x1, eps_times=eps_times, n_iter=n_iter)
print(f"done. P.shape={P.shape}, P.sum()={P.sum():.6f}")

# --- save ---
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"ot_plan_toy_n0{n0}_n1{n1}_seed{seed}_epstimes{eps_times}.npz")
np.savez(out_path, P=P, x0=x0, x1=x1,
         n0=n0, n1=n1, seed=seed, eps_times=eps_times)
print(f"saved to {out_path}")
