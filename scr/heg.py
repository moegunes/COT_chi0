import numba
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


@numba.njit(cache=True, inline="always")
def chiReal_numba(v0, i, j, rx, ry, rz, R_list, mu=0):
    pi = np.pi
    n_R = len(R_list)
    N = len(rx)
    kF = np.sqrt(-2.0 * (v0 - 1e-10))
    NF = kF / (2.0 * pi * pi)
    n_dens = kF * kF * kF / (6.0 * pi * pi)
    factor = 12.0 * pi * n_dens * NF * 2.0
    kF2 = 2.0 * kF

    rxi = rx[i]
    ryi = ry[i]
    rzi = rz[i]

    # Pre-compute abs differences (thread-local arrays)
    vx = abs(rx[j] - rxi)
    vy = abs(ry[j] - ryi)
    vz = abs(rz[j] - rzi)

    chi_val = 0
    for k in range(n_R):
        Rx = R_list[k, 0]
        Ry = R_list[k, 1]
        Rz = R_list[k, 2]
        ddx = vx - Rx
        ddy = vy - Ry
        ddz = vz - Rz
        r = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
        u = kF2 * r
        denom = kF2 * (r + 1e-15)
        denom2 = denom * denom
        denom4 = denom2 * denom2

        chi_val += -factor * (np.sin(u) - u * np.cos(u)) / denom4
    return chi_val


def n_h(vh):
    kF = (-2 * vh) ** (1 / 2)
    return kF**3 / (3 * np.pi**2)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _compute_COT1_numba(V_ksR, rx, ry, rz, R_list, indices):
    """Numba-compiled parallel COT1 density computation."""

    n_pts = len(indices)
    N = len(V_ksR)
    n_R = len(R_list)
    results = np.empty(n_pts)
    pi = np.pi

    for idx in numba.prange(n_pts):
        i = indices[idx]
        v0 = V_ksR[i]

        kF = np.sqrt(-2.0 * (v0 - 1e-10))
        NF = kF / (2.0 * pi * pi)
        n_dens = kF * kF * kF / (6.0 * pi * pi)
        factor = 12.0 * pi * n_dens * NF * 2.0
        kF2 = 2.0 * kF

        rxi = rx[i]
        ryi = ry[i]
        rzi = rz[i]

        # Pre-compute abs differences (thread-local arrays)
        vx = np.empty(N)
        vy = np.empty(N)
        vz = np.empty(N)
        for j in range(N):
            vx[j] = abs(rx[j] - rxi)
            vy[j] = abs(ry[j] - ryi)
            vz[j] = abs(rz[j] - rzi)

        chi_V_sum = 0.0
        chi_sum = 0.0

        for j in range(N):
            chi_val = 0
            for k in range(n_R):
                Rx = R_list[k, 0]
                Ry = R_list[k, 1]
                Rz = R_list[k, 2]
                ddx = vx[j] - Rx
                ddy = vy[j] - Ry
                ddz = vz[j] - Rz
                r = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                u = kF2 * r
                denom = kF2 * (r + 1e-15)
                denom2 = denom * denom
                denom4 = denom2 * denom2

                chi_val += -factor * (np.sin(u) - u * np.cos(u)) / denom4
            chi_V_sum += chi_val * V_ksR[j]
            chi_sum += chi_val

        Vcon = chi_V_sum / chi_sum
        kF_c = np.sqrt(-2.0 * Vcon)
        results[idx] = kF_c * kF_c * kF_c / (3.0 * pi * pi)

    return results
