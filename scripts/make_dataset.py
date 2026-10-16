"""Step 2: generate a dataset by running the FE solver over a range of loads and stiffnesses.

Parameters are drawn by Latin hypercube sampling:

    P      tip force             -50 ... 50  kN
    q      distributed load      -20 ... 20  kN/m
    E      Young's modulus        50 ... 250 GPa  (log-uniform: aluminium to steel)
    taper  area reduction          0 ... 0.5

Each sample stores the FE nodal displacements and the exact solution at the same
nodes. A separate out-of-range set (E = 250-400 GPa) tests extrapolation.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barfe import Bar, analytic, fem  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_ELEMENTS = 40
N_SAMPLES = 6000
N_OOD = 500

RANGES = {
    "P": (-50e3, 50e3),
    "q": (-20e3, 20e3),
    "log10_E": (np.log10(50e9), np.log10(250e9)),
    "taper": (0.0, 0.5),
}


def sample(n, ranges, seed):
    lo = np.array([r[0] for r in ranges.values()])
    hi = np.array([r[1] for r in ranges.values()])
    unit = qmc.LatinHypercube(d=len(ranges), seed=seed).random(n)
    s = qmc.scale(unit, lo, hi)
    return np.column_stack([s[:, 0], s[:, 1], 10 ** s[:, 2], s[:, 3]])  # P, q, E, taper


def run(params):
    u_fe, u_ex, x = [], [], None
    t0 = time.perf_counter()
    for P, q, E, taper in params:
        bar = Bar(E=E, P=P, q=q, taper=taper)
        res = fem.solve(bar, N_ELEMENTS)
        u_fe.append(res.u)
        u_ex.append(analytic.displacement(bar, res.x))
        x = res.x
    dt = time.perf_counter() - t0
    return x, np.array(u_fe), np.array(u_ex), dt


def main():
    (ROOT / "data").mkdir(exist_ok=True)

    params = sample(N_SAMPLES, RANGES, seed=0)
    x, u_fe, u_ex, dt = run(params)
    print(f"{N_SAMPLES} FE solves in {dt:.2f} s ({dt / N_SAMPLES * 1e3:.3f} ms each)")

    ood_ranges = dict(RANGES, log10_E=(np.log10(250e9), np.log10(400e9)))
    params_ood = sample(N_OOD, ood_ranges, seed=1)
    _, u_fe_ood, u_ex_ood, _ = run(params_ood)

    np.savez_compressed(
        ROOT / "data" / "dataset.npz",
        x=x, params=params, u_fe=u_fe, u_exact=u_ex,
        params_ood=params_ood, u_fe_ood=u_fe_ood, u_exact_ood=u_ex_ood,
        fe_seconds_per_solve=dt / N_SAMPLES,
        param_names=np.array(["P_N", "q_N_per_m", "E_Pa", "taper"]),
    )
    print(f"Tip displacement range: {u_fe[:, -1].min() * 1e3:.3f} to {u_fe[:, -1].max() * 1e3:.3f} mm")
    print(f"Max FE-vs-exact nodal error over dataset: {np.abs(u_fe - u_ex).max():.2e} m")
    print(f"Saved data/dataset.npz ({N_SAMPLES} samples + {N_OOD} out-of-range)")


if __name__ == "__main__":
    main()
