"""
xc.py — Exchange-correlation functionals
==========================================

LDA exchange-correlation potential using Perdew-Zunger parametrization
for the correlation part. Pure functions, no classes.
"""

import numpy as np


def _pz_correlation_potential(n):
    """Perdew-Zunger parametrization of the LDA correlation potential.

    Parameters
    ----------
    n : np.ndarray
        Electron density (must be positive).

    Returns
    -------
    np.ndarray
        Correlation potential v_c(n).
    """
    rs = (4 * np.pi / 3 * n) ** (-1 / 3)

    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    Au = 0.0311
    Bu = -0.048
    Cu = 0.0020
    Du = -0.0116

    mask_low = rs < 1
    mask_high = rs >= 1

    # rs < 1 regime
    v_low = (
        Au * np.log(rs)
        + (Bu - 1 / 3 * Au)
        + 2 / 3 * Cu * rs * np.log(rs)
        + 1 / 3 * (2 * Du - Cu) * rs
    )

    # rs >= 1 regime
    v_cep = gamma / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
    v_high = (
        v_cep
        * (1 + 7 / 6 * beta1 * np.sqrt(rs) + 4 / 3 * beta2 * rs)
        / (1 + beta1 * np.sqrt(rs) + beta2 * rs)
    )

    return v_low * mask_low + v_high * mask_high


def lda_vxc(densR):
    """Compute LDA exchange-correlation potential in real space.

    V_xc(r) = V_x(r) + V_c(r)
    where V_x = -(3/π)^(1/3) * n^(1/3) (Slater exchange)
    and V_c is Perdew-Zunger parametrization.

    Parameters
    ----------
    densR : np.ndarray
        Electron density in real space.

    Returns
    -------
    np.ndarray
        Exchange-correlation potential V_xc(r).
    """
    n = np.abs(densR.real)
    n = np.maximum(n, 1e-30)  # avoid division by zero

    v_x = -((3.0 / np.pi) ** (1.0 / 3.0)) * n ** (1.0 / 3.0)
    v_c = _pz_correlation_potential(n)

    return v_x + v_c


def normalize_density(densR, n_electrons, dvol):
    """Shift density so it integrates to the correct electron number.

    Used for V_xc calculation only (not for the output density).
    When the shift would produce negative densities, those are set to zero
    and the shift is re-adjusted via bisection.

    Parameters
    ----------
    densR : np.ndarray
        Electron density in real space.
    n_electrons : float
        Target number of electrons.
    dvol : float
        Volume element (dx^3).

    Returns
    -------
    np.ndarray
        Shifted density with correct electron number.
    """
    rho = np.asarray(densR.real, dtype=float)
    N_current = np.sum(rho) * dvol

    if N_current <= n_electrons:
        # Add charge uniformly
        shift = (n_electrons - N_current) / (rho.size * dvol)
        rho_shifted = rho + shift
    else:
        # Remove charge: bisect to find shift such that
        # sum(max(rho - shift, 0)) * dvol = n_electrons
        lo, hi = 0.0, np.max(rho)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            N_mid = np.sum(np.maximum(rho - mid, 0.0)) * dvol
            if N_mid > n_electrons:
                lo = mid
            else:
                hi = mid
        alpha = 0.5 * (lo + hi)
        rho_shifted = np.maximum(rho - alpha, 0.0)

    return np.maximum(rho_shifted, 1e-12)
