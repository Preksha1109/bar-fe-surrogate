"""Problem definition for an axially loaded, linearly tapered bar.

    x = 0                                            x = L
    |#|=================================================>  P (tip force)
    |#|  ->  ->  ->  ->  ->  ->  ->  ->  ->  ->  ->  ->      q (load per unit length)
    fixed                                            free

Governing equation (linear elasticity, small strain):

    d/dx ( E A(x) du/dx ) + q = 0,    u(0) = 0,    E A(L) u'(L) = P

with a linearly tapered cross-section A(x) = A0 * (1 - taper * x / L).
SI units throughout (N, m, Pa).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Bar:
    E: float              # Young's modulus [Pa]
    P: float              # axial tip force at x = L [N] (positive = tension)
    q: float = 0.0        # uniform distributed axial load [N/m]
    taper: float = 0.0    # A(L) = A0 * (1 - taper); 0 <= taper < 1
    L: float = 1.0        # length [m]
    A0: float = 4e-4      # root cross-sectional area [m^2] (20 mm x 20 mm)

    def __post_init__(self):
        if not 0.0 <= self.taper < 1.0:
            raise ValueError("taper must satisfy 0 <= taper < 1")
        if self.E <= 0 or self.L <= 0 or self.A0 <= 0:
            raise ValueError("E, L and A0 must be positive")

    def area(self, x):
        return self.A0 * (1.0 - self.taper * x / self.L)

    def axial_force(self, x):
        """Internal axial force N(x) from equilibrium of the free end segment."""
        return self.P + self.q * (self.L - x)
