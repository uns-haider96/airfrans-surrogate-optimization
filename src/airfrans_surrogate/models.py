"""Network architectures used across the phases."""

import torch.nn as nn


class PointwiseMLP(nn.Module):
    """Field surrogate applied independently at every mesh node (Phases 1 and 1b)."""

    def __init__(self, n_in, n_out, width, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ForceMLP(nn.Module):
    """Direct force surrogate: shape descriptor, incidence, Reynolds number -> (CL, log CD) (Phase 2b)."""

    def __init__(self, n_in, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, 2))

    def forward(self, x):
        return self.net(x)
