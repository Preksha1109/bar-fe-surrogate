"""Small MLP surrogate: (P, q, E, taper) -> nodal displacements.

Inputs are scaled to roughly [-1, 1] (E on a log scale). Outputs are the
displacements at the free nodes in mm, standardised per node. The fixed node
at x = 0 is always zero and is not predicted.
"""

import numpy as np
import torch
from torch import nn

P_SCALE = 50e3
Q_SCALE = 20e3
LOG10_E_CENTRE = 0.5 * (np.log10(50e9) + np.log10(250e9))
LOG10_E_HALF = 0.5 * (np.log10(250e9) - np.log10(50e9))


def encode_inputs(params: np.ndarray) -> np.ndarray:
    P, q, E, taper = params.T
    return np.column_stack([
        P / P_SCALE,
        q / Q_SCALE,
        (np.log10(E) - LOG10_E_CENTRE) / LOG10_E_HALF,
        (taper - 0.25) / 0.25,
    ]).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, n_out: int, width: int = 64, depth: int = 3):
        super().__init__()
        layers, n_in = [], 4
        for _ in range(depth):
            layers += [nn.Linear(n_in, width), nn.SiLU()]
            n_in = width
        layers.append(nn.Linear(n_in, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Surrogate:
    """Wraps the network with its output scaling so predictions come out in metres."""

    def __init__(self, model: MLP, u_mean: np.ndarray, u_std: np.ndarray):
        self.model = model.eval()
        self.u_mean = u_mean.astype(np.float32)  # per free node, mm
        self.u_std = u_std.astype(np.float32)

    @torch.no_grad()
    def predict(self, params: np.ndarray) -> np.ndarray:
        z = self.model(torch.from_numpy(encode_inputs(np.atleast_2d(params)))).numpy()
        u_free_mm = z * self.u_std + self.u_mean
        u = np.zeros((len(u_free_mm), u_free_mm.shape[1] + 1))
        u[:, 1:] = u_free_mm * 1e-3
        return u

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(),
                    "n_out": self.model.net[-1].out_features,
                    "width": self.model.net[0].out_features,
                    "depth": (len(self.model.net) - 1) // 2,
                    "u_mean": self.u_mean, "u_std": self.u_std}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        model = MLP(ck["n_out"], ck["width"], ck["depth"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, ck["u_mean"], ck["u_std"])
