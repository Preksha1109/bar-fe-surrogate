"""Step 1: check the FE solver against the closed-form solution.

Prints the nodal error for a prismatic bar (should be round-off only) and a
mesh-convergence study for a tapered bar (should show second-order convergence).
Writes results/convergence.json for the README plot.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barfe import Bar, analytic, fem  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def max_nodal_error(bar, n):
    res = fem.solve(bar, n_elements=n)
    return np.abs(res.u - analytic.displacement(bar, res.x)).max()


def main():
    OUT.mkdir(exist_ok=True)

    prismatic = Bar(E=200e9, P=40e3, q=15e3, taper=0.0)
    e_pr = max_nodal_error(prismatic, 40)
    print(f"Prismatic bar, 40 elements: max nodal error = {e_pr:.2e} m "
          f"(tip u = {analytic.tip_displacement(prismatic) * 1e3:.4f} mm)")

    tapered = Bar(E=200e9, P=40e3, q=15e3, taper=0.5)
    u_tip = analytic.tip_displacement(tapered)
    ns = [5, 10, 20, 40, 80, 160, 320]
    rows = []
    print(f"\nTapered bar (taper 0.5), exact tip u = {u_tip * 1e3:.6f} mm")
    print(f"{'elements':>9} {'h [m]':>9} {'max err [m]':>12} {'rel to tip':>11} {'rate':>6}")
    prev = None
    for n in ns:
        e = max_nodal_error(tapered, n)
        rate = np.log2(prev / e) if prev else float("nan")
        print(f"{n:>9d} {1 / n:>9.4f} {e:>12.3e} {e / abs(u_tip):>11.2e} {rate:>6.2f}")
        rows.append({"n_elements": n, "h": tapered.L / n, "max_error_m": e})
        prev = e

    res = fem.solve(tapered, 40)
    print(f"\nSupport reaction: {res.reaction:.3f} N "
          f"(expected {-(tapered.P + tapered.q * tapered.L):.3f} N)")

    (OUT / "convergence.json").write_text(json.dumps({
        "prismatic_max_error_m": e_pr,
        "tapered_case": {"E": tapered.E, "P": tapered.P, "q": tapered.q,
                         "taper": tapered.taper, "tip_u_m": u_tip},
        "study": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
