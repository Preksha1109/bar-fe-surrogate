"""Linear (2-node) finite element solver for the axially loaded bar.

Steps, kept separate so each can be inspected and tested:

    mesh -> element stiffness -> global assembly -> load vector
         -> boundary conditions -> solve -> nodal displacements
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problem import Bar


@dataclass
class FEResult:
    x: np.ndarray        # nodal coordinates [m]
    u: np.ndarray        # nodal displacements [m]
    K: sp.csr_matrix     # assembled global stiffness (before BCs)
    F: np.ndarray        # global load vector (before BCs)
    reaction: float      # support reaction at x = 0 [N]

    def element_strain(self):
        return np.diff(self.u) / np.diff(self.x)


def mesh(L: float, n_elements: int):
    """Uniform mesh: node coordinates and element connectivity."""
    x = np.linspace(0.0, L, n_elements + 1)
    conn = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    return x, conn


def element_stiffness(E: float, A: float, h: float):
    """2x2 stiffness of a linear bar element with constant EA over its length."""
    return (E * A / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])


def assemble_stiffness(bar: Bar, x: np.ndarray, conn: np.ndarray):
    """Global stiffness matrix. Each element uses the area at its midpoint,
    which for a linear taper equals the element's mean area."""
    n_nodes = len(x)
    rows, cols, vals = [], [], []
    for i, j in conn:
        h = x[j] - x[i]
        ke = element_stiffness(bar.E, bar.area(0.5 * (x[i] + x[j])), h)
        dofs = (i, j)
        for a in range(2):
            for b in range(2):
                rows.append(dofs[a])
                cols.append(dofs[b])
                vals.append(ke[a, b])
    # COO -> CSR sums duplicate entries, which is exactly the assembly operation.
    return sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()


def assemble_loads(bar: Bar, x: np.ndarray, conn: np.ndarray):
    """Consistent nodal loads for the uniform load q plus the point load P at the tip."""
    F = np.zeros(len(x))
    for i, j in conn:
        h = x[j] - x[i]
        F[i] += 0.5 * bar.q * h
        F[j] += 0.5 * bar.q * h
    F[-1] += bar.P
    return F


def solve(bar: Bar, n_elements: int = 40) -> FEResult:
    x, conn = mesh(bar.L, n_elements)
    K = assemble_stiffness(bar, x, conn)
    F = assemble_loads(bar, x, conn)

    # Dirichlet BC u(0) = 0 by elimination: drop the fixed DOF's row and column.
    free = np.arange(1, len(x))
    u = np.zeros(len(x))
    u[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])

    reaction = float((K @ u)[0] - F[0])
    return FEResult(x=x, u=u, K=K, F=F, reaction=reaction)
