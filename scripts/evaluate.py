"""Step 4: compare surrogate and FE errors against the exact solution, and draw the README figure.

All errors are measured against the closed-form solution at the 41 FE nodes:
  * FE error   = discretisation error of the 40-element mesh
  * NN error   = surrogate error (it was trained on FE results, so it also inherits the FE error)
  * NN vs FE   = how well the network reproduces the solver it was trained on
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from barfe import Bar, analytic  # noqa: E402
from barfe.surrogate import Surrogate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Reference palette: slot 1 (blue) = FE, slot 2 (orange) = NN, text ink = exact.
C_FE, C_NN = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"


def error_stats(u_pred, u_ref, scale):
    err = u_pred - u_ref
    per_sample_max = np.abs(err).max(axis=1)
    rel_l2 = np.linalg.norm(err, axis=1) / np.maximum(np.linalg.norm(u_ref, axis=1), 1e-12)
    return {
        "max_abs_um": float(np.abs(err).max() * 1e6),
        "rms_um": float(np.sqrt((err ** 2).mean()) * 1e6),
        "rms_pct_of_rms_disp": float(np.sqrt((err ** 2).mean()) / scale * 100),
        "median_rel_l2_pct": float(np.median(rel_l2) * 100),
        "p95_rel_l2_pct": float(np.percentile(rel_l2, 95) * 100),
    }, per_sample_max


def time_nn(sur, params, repeats=20):
    t0 = time.perf_counter()
    for _ in range(repeats):
        sur.predict(params)
    batch = (time.perf_counter() - t0) / repeats / len(params)
    t0 = time.perf_counter()
    for p in params[:200]:
        sur.predict(p)
    single = (time.perf_counter() - t0) / 200
    return batch, single


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def main():
    d = np.load(ROOT / "data" / "dataset.npz")
    split = np.load(ROOT / "results" / "split.npz")
    te = split["test"]
    x = d["x"]
    params, u_fe, u_ex = d["params"][te], d["u_fe"][te], d["u_exact"][te]
    p_ood, u_fe_ood, u_ex_ood = d["params_ood"], d["u_fe_ood"], d["u_exact_ood"]

    sur = Surrogate.load(ROOT / "models" / "surrogate.pt")
    u_nn, u_nn_ood = sur.predict(params), sur.predict(p_ood)

    scale = np.sqrt((u_ex ** 2).mean())
    fe_stats, fe_max = error_stats(u_fe, u_ex, scale)
    nn_stats, nn_max = error_stats(u_nn, u_ex, scale)
    nnfe_stats, _ = error_stats(u_nn, u_fe, scale)
    ood_scale = np.sqrt((u_ex_ood ** 2).mean())
    nn_ood_stats, nn_ood_max = error_stats(u_nn_ood, u_ex_ood, ood_scale)
    fe_ood_stats, fe_ood_max = error_stats(u_fe_ood, u_ex_ood, ood_scale)

    nn_batch, nn_single = time_nn(sur, params)
    fe_time = float(d["fe_seconds_per_solve"])

    metrics = {
        "test_samples": int(len(te)),
        "rms_displacement_um": float(scale * 1e6),
        "fe_vs_exact": fe_stats,
        "nn_vs_exact": nn_stats,
        "nn_vs_fe": nnfe_stats,
        "out_of_range_E_250_400GPa": {"fe_vs_exact": fe_ood_stats, "nn_vs_exact": nn_ood_stats},
        "timing_ms_per_sample": {"fe_solve": fe_time * 1e3, "nn_batched": nn_batch * 1e3,
                                 "nn_single_call": nn_single * 1e3},
    }
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Test set: {len(te)} samples, RMS displacement {scale * 1e6:.1f} um")
    print(f"{'':22}{'max |err| um':>13}{'RMS um':>9}{'RMS %':>8}{'med relL2 %':>12}{'p95 relL2 %':>12}")
    for name, s in [("FE (40 el) vs exact", fe_stats), ("NN vs exact", nn_stats),
                    ("NN vs FE", nnfe_stats), ("NN vs exact, E>250GPa", nn_ood_stats)]:
        print(f"{name:22}{s['max_abs_um']:13.3f}{s['rms_um']:9.3f}{s['rms_pct_of_rms_disp']:8.3f}"
              f"{s['median_rel_l2_pct']:12.4f}{s['p95_rel_l2_pct']:12.3f}")
    print(f"Timing per sample: FE {fe_time * 1e3:.3f} ms | NN batched {nn_batch * 1e3:.4f} ms"
          f" | NN single call {nn_single * 1e3:.3f} ms")

    # ------------------------------------------------------------------ figure
    conv = json.loads((ROOT / "results" / "convergence.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), facecolor="#fcfcfb")
    for ax in axes.flat:
        ax.set_facecolor("#fcfcfb")
        style(ax)

    # (a) mesh convergence
    ax = axes[0, 0]
    h = np.array([r["h"] for r in conv["study"]])
    e = np.array([r["max_error_m"] for r in conv["study"]]) * 1e6
    ax.loglog(h, e, "-o", color=C_FE, lw=2, ms=6, label="FE max nodal error")
    ref = 0.4 * e[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "--", color=MUTED, lw=1.2, label="slope 2 reference")
    i40 = [r["n_elements"] for r in conv["study"]].index(40)
    ax.annotate("40 elements\n(used for dataset)", (h[i40], e[i40]), xytext=(12, -28),
                textcoords="offset points", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlabel("element length h [m]")
    ax.set_ylabel("max nodal error [µm]")
    ax.set_title("(a) FE converges to the exact solution at O(h²)", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # (b) example displacement fields
    ax = axes[0, 1]
    tip = u_ex[:, -1]
    curved = np.where((tip > 0.5e-3) & (tip < 1.5e-3) & (params[:, 1] < -12e3))[0]
    picks = [int(np.argmax(tip)), int(curved[np.argmax(params[curved, 0])]), int(np.argmin(tip))]
    xf = np.linspace(0, 1, 200)
    case_lines = []
    for k, i in enumerate(picks):
        P, q, E, taper = params[i]
        bar = Bar(E=E, P=P, q=q, taper=taper)
        ax.plot(xf, analytic.displacement(bar, xf) * 1e3, color=INK, lw=1.4,
                label="exact" if k == 0 else None, zorder=1)
        ax.plot(x[::4], u_fe[i, ::4] * 1e3, "o", color=C_FE, ms=6, mec="#fcfcfb", mew=1.2,
                label="FE (every 4th node)" if k == 0 else None, zorder=2)
        ax.plot(x, u_nn[i] * 1e3, "--", color=C_NN, lw=2, label="NN" if k == 0 else None, zorder=3)
        letter = "ABC"[k]
        ax.annotate(letter, (1.0, u_ex[i, -1] * 1e3), xytext=(8, -4), textcoords="offset points",
                    fontsize=9.5, fontweight="bold", color=INK)
        case_lines.append(f"{letter}: P={P / 1e3:+.0f} kN, q={q / 1e3:+.0f} kN/m, "
                          f"E={E / 1e9:.0f} GPa, taper={taper:.2f}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-0.02, 1.08)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("displacement u [mm]")
    ax.set_title("(b) Test cases: exact vs FE vs NN", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(0.02, 0.03, "\n".join(case_lines), transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=INK2, linespacing=1.5)

    # (c) error distribution on the test set
    ax = axes[1, 0]
    bins = np.logspace(np.log10(min(fe_max.min(), nn_max.min()) * 0.8),
                       np.log10(max(fe_max.max(), nn_max.max()) * 1.2), 45)
    ax.hist(fe_max * 1e6, bins=bins * 1e6, color=C_FE, alpha=0.85, label="FE (40 elements)",
            edgecolor="#fcfcfb", linewidth=0.5)
    ax.hist(nn_max * 1e6, bins=bins * 1e6, color=C_NN, alpha=0.85, label="NN surrogate",
            edgecolor="#fcfcfb", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("max nodal error vs exact, per test sample [µm]")
    ax.set_ylabel("number of samples")
    ax.set_title(f"(c) Error distribution, {len(te)} test samples", loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    # (d) error vs stiffness, including outside the training range
    ax = axes[1, 1]
    ax.axvspan(50, 250, color="#f0efec", zorder=0)
    E_all = np.concatenate([params[:, 2], p_ood[:, 2]]) / 1e9
    ax.scatter(E_all, np.concatenate([fe_max, fe_ood_max]) * 1e6, s=10, color=C_FE, alpha=0.6,
               lw=0, label="FE")
    ax.scatter(E_all, np.concatenate([nn_max, nn_ood_max]) * 1e6, s=10, color=C_NN, alpha=0.6,
               lw=0, label="NN")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([50, 100, 200, 400])
    ax.set_xticklabels(["50", "100", "200", "400"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Young's modulus E [GPa]")
    ax.set_ylabel("max nodal error vs exact [µm]")
    ax.set_title("(d) NN degrades outside its training range; FE does not", loc="left",
                 fontsize=10.5, color=INK)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(matplotlib.patches.Patch(color="#f0efec", label="training range of E"))
    ax.legend(handles=handles, fontsize=8.5, loc="lower left", markerscale=2,
              facecolor="#fcfcfb", edgecolor=GRID, framealpha=0.95)

    fig.tight_layout(h_pad=2.2, w_pad=2.0)
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "comparison.png", dpi=150)
    print("Saved docs/comparison.png")


if __name__ == "__main__":
    main()
