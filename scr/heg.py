import numpy as np


def chiReal(v0, i, grid, mu=0):
    "Lindhard function (real space) from Vignale, eqn. 19 in the report"
    rlist = grid.rlist
    MtR = grid.MtR
    rvec = np.array(rlist)
    vec = np.abs(rvec - rvec[i])
    # Safety: ensure v0 is negative enough for sqrt
    # v0_safe = np.minimum(v0, -1e-10)
    kF = (-2 * (v0 - 1e-10)) ** (1 / 2)
    NF = kF / (2 * np.pi**2)
    n = kF**3 / (6 * np.pi**2)  # HEG density
    factor = 12 * np.pi * n * NF * 2  # prefactor
    # for a periodic system, we use periodic Lindhard function which is added for all unit cells
    k = 2  # number of unit cells to be looped over (2 is enough to converge)
    res = np.zeros(len(rlist))
    for a in range(0, k):
        for b in range(0, k):
            for c in range(0, k):
                R = a * MtR[0] + b * MtR[1] + c * MtR[2]
                r = np.linalg.norm(vec - R, axis=1)
                res += (
                    -factor
                    * (np.sin(2 * kF * (r)) - 2 * kF * r * np.cos(2 * kF * (r)))
                    / (2 * kF * (r + 1e-15)) ** 4
                )
    return res


def n_h(vh):
    kF = (-2 * vh) ** (1 / 2)
    return kF**3 / (3 * np.pi**2)
