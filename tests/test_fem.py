import numpy as np
import pytest

from barfe import Bar, analytic, fem


def test_stiffness_matrix_properties():
    bar = Bar(E=200e9, P=10e3, taper=0.4)
    x, conn = fem.mesh(bar.L, 10)
    K = fem.assemble_stiffness(bar, x, conn).toarray()
    assert np.allclose(K, K.T)
    # Rigid-body translation produces no force: rows sum to zero, K is singular.
    assert np.allclose(K.sum(axis=1), 0.0, atol=1e-6 * np.abs(K).max())
    assert np.linalg.matrix_rank(K) == len(x) - 1
    # Tridiagonal for a 1D chain of elements.
    assert np.allclose(np.triu(K, 2), 0.0)


@pytest.mark.parametrize("P, q", [(10e3, 0.0), (0.0, 25e3), (-30e3, 12e3)])
def test_prismatic_bar_is_nodally_exact(P, q):
    # For constant EA, linear elements with consistent loads reproduce the
    # exact solution at the nodes regardless of mesh size.
    bar = Bar(E=70e9, P=P, q=q, taper=0.0)
    res = fem.solve(bar, n_elements=7)
    u_exact = analytic.displacement(bar, res.x)
    assert np.allclose(res.u, u_exact, rtol=1e-10, atol=1e-15)


def test_tip_load_textbook_value():
    bar = Bar(E=200e9, P=50e3, taper=0.0)
    assert analytic.tip_displacement(bar) == pytest.approx(bar.P * bar.L / (bar.E * bar.A0))


def test_reaction_balances_applied_load():
    bar = Bar(E=110e9, P=20e3, q=-8e3, taper=0.3)
    res = fem.solve(bar, n_elements=20)
    assert res.reaction == pytest.approx(-(bar.P + bar.q * bar.L), rel=1e-9)


def test_tapered_bar_converges_at_second_order():
    bar = Bar(E=200e9, P=40e3, q=15e3, taper=0.5)
    errs, hs = [], []
    for n in (8, 16, 32, 64, 128):
        res = fem.solve(bar, n_elements=n)
        errs.append(np.abs(res.u - analytic.displacement(bar, res.x)).max())
        hs.append(bar.L / n)
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.05)


def test_analytic_series_and_log_branches_agree():
    x = np.linspace(0, 1, 11)
    just_below = Bar(E=200e9, P=1e4, q=5e3, taper=0.99e-3)
    just_above = Bar(E=200e9, P=1e4, q=5e3, taper=1.01e-3)
    u_lo = analytic.displacement(just_below, x)
    u_hi = analytic.displacement(just_above, x)
    # Fields differ only by the tiny taper change, so they should be close.
    assert np.allclose(u_lo, u_hi, rtol=1e-4)
    # And the log branch matches FE on a fine mesh.
    res = fem.solve(just_above, n_elements=400)
    assert np.allclose(res.u, analytic.displacement(just_above, res.x), rtol=1e-8)
