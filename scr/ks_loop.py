"""
ks_loop.py — Kohn-Sham self-consistent loop using COT approximations
=====================================================================

Replaces the KS loop from tmp_cot_ks.py (LPA=2, LPA=4 in KS_solver).
No classes — pure functions compatible with numba inner kernels.

The loop:
    1. Start from initial density (uniform or provided)
    2. Build V_KS = V_ext + V_H[n] + V_xc[n]
    3. Compute density from V_KS using a COT approximation
    4. Mix new and old density
    5. Repeat until convergence

Usage
-----
    from scr.ks_loop import run_ks_loop
    density, history = run_ks_loop(system, grid, approximation=COT1_AV,
                                    n_iter=100, mixing=0.65, n_electrons=2)
"""

import time

import numpy as np

from .fourier import fft_g_to_r, fft_r_to_g
from .xc import lda_vxc, normalize_density


def _build_vks(
    densR,
    V_extR,
    G_dens_int,
    HP_nodens,
    n_rgrid,
    n_electrons=None,
    dvol=None,
    shift_density=False,
    gspace=True,
):
    """Build the Kohn-Sham potential from density.

    V_KS(r) = V_ext(r) + V_H(r) + V_xc(r)

    When gspace=True (default), all components are summed in G-space and
    then transformed to R-space together. This matches the old code's
    approach (LPA=2/4) and ensures consistency with the energy cutoff.

    When gspace=False (COT0/LPA), components are summed directly in R-space
    (only V_H goes through G-space). This preserves high-frequency V_xc
    components.

    Parameters
    ----------
    densR : np.ndarray
        Current density in real space.
    V_extR : np.ndarray
        External potential in real space.
    G_dens_int : np.ndarray
        Integer G-vector indices.
    HP_nodens : np.ndarray
        Hartree prefactors: 4pi/|G|^2 (0 for G=0).
    n_rgrid : int
        Grid size per dimension.
    n_electrons : float, optional
        Target number of electrons (needed when shift_density=True).
    dvol : float, optional
        Volume element dx^3 (needed when shift_density=True).
    shift_density : bool
        If True, shift density for V_xc to conserve electron number
        (paper Sec. V). If False, use raw density (default).
    gspace : bool
        If True, sum V_ext + V_xc + V_H in G-space (matches old LPA=2/4).
        If False, sum in R-space (matches old LPA=1).

    Returns
    -------
    np.ndarray
        V_KS in real space (real-valued).
    """
    # Density in G-space (always needed for Hartree)
    densG = fft_r_to_g(densR, G_dens_int, n_rgrid)
    V_H_G = densG * HP_nodens

    # XC potential: from shifted density if requested, otherwise actual
    if shift_density and n_electrons is not None and dvol is not None:
        dens_for_vxc = normalize_density(densR, n_electrons, dvol)
    else:
        dens_for_vxc = np.abs(densR.real)
    V_xc_R = lda_vxc(dens_for_vxc)

    if gspace:
        # G-space approach: sum all components in G-space, then transform
        V_ext_G = fft_r_to_g(V_extR, G_dens_int, n_rgrid)
        V_xc_G = fft_r_to_g(V_xc_R, G_dens_int, n_rgrid)
        V_ks_G = V_ext_G + V_xc_G + V_H_G
        V_ksR = fft_g_to_r(V_ks_G, G_dens_int, n_rgrid).real
    else:
        # R-space approach: only V_H goes through G-space
        V_H_R = fft_g_to_r(V_H_G, G_dens_int, n_rgrid).real
        V_ksR = (V_extR + V_H_R + V_xc_R).real

    return V_ksR


def run_ks_loop(
    system,
    grid,
    approximation,
    n_iter=100,
    mixing=0.65,
    n_electrons=2,
    densR_init=None,
    convergence_threshold=None,
    full_grid=True,
    shift_density=False,
):
    """Run the Kohn-Sham self-consistent loop with a COT density functional.

    Instead of solving the KS Schroedinger equation, uses a COT approximation
    (e.g., COT1-av) to compute the density from V_KS at each iteration.

    Parameters
    ----------
    system : MaterialConfig
        Material configuration with V_extR.
    grid : Grid
        Spatial grid data.
    approximation : int
        COT approximation type for the density step (e.g., COT1_AV=2).
    n_iter : int
        Maximum number of self-consistent iterations.
    mixing : float
        Linear mixing: n = mixing * n_COT + (1-mixing) * n_old.
    n_electrons : float
        Number of electrons (2 for He).
    densR_init : np.ndarray, optional
        Initial density. If None, uses uniform density with correct N_el.
    convergence_threshold : float, optional
        Stop when integral |n_new - n_old| dr < threshold.
    full_grid : bool
        Compute density on full grid (True) or trajectory only (False).
    shift_density : bool
        If True, shift density for V_xc to conserve electron number
        (paper Sec. V modified SC-COT). If False, use raw density.

    Returns
    -------
    densR : np.ndarray
        Final converged density.
    history : list of np.ndarray
        Density at each iteration (including initial).
    """
    # Lazy import to avoid circular dependency
    from .cot import COT0, _get_dens_COT1_fast
    from .heg import n_h

    n = grid.n_rgrid
    N = n**3
    G_dens_int = grid.G_dens_int
    HP_nodens = grid.HP_nodens
    dvol = (system.a_l / n) ** 3

    # Initial density
    if densR_init is not None:
        densR = np.array(densR_init, dtype=float)
    else:
        # Uniform density with correct electron number
        V_cell = system.a_l**3
        densR = np.ones(N) * (n_electrons / V_cell)

    history = [densR.copy()]

    print(f"Starting KS loop: {n_iter} iterations, mixing={mixing}")
    print(f"Initial N_el = {np.sum(densR) * dvol:.4f}")
    if shift_density:
        print(f"Density shift enabled (target N_el={n_electrons})")
    total_start = time.perf_counter()

    for it in range(1, n_iter + 1):
        iter_start = time.perf_counter()

        # 1. Build V_KS from current density
        #    COT0 (LPA): R-space approach (matches old LPA=1)
        #    Connectors: G-space approach (matches old LPA=2/4)
        use_gspace = approximation != COT0
        V_ksR = _build_vks(
            densR,
            system.V_extR,
            G_dens_int,
            HP_nodens,
            n,
            n_electrons=n_electrons,
            dvol=dvol,
            shift_density=shift_density,
            gspace=use_gspace,
        )

        # 2. Shift V_KS if it has positive values (only for connector, not COT0)
        #    For COT0 (LPA), positive V_KS gives n_h=0 naturally (kF is imaginary).
        #    For connectors (COT1, COT1-av), V_KS must be all negative.
        if approximation != COT0 and V_ksR.max() > 0:
            print("  V_KS exceeds zero. Shifting...")
            V_ksR = V_ksR - V_ksR.max()

        # 3. Compute new density using COT approximation
        #    Temporarily set system.V_ksR for the COT computation
        V_ksR_backup = system.V_ksR
        system.V_ksR = V_ksR

        if approximation == COT0:
            # For COT0, density is 0 where V_KS > 0 (no electrons above Fermi level)
            mask = V_ksR < 0
            densR_new = np.zeros_like(V_ksR)
            densR_new[mask] = n_h(V_ksR[mask])
        else:
            densR_new = _get_dens_COT1_fast(approximation, system, grid, full_grid)

        system.V_ksR = V_ksR_backup

        # 4. Mix densities
        densR = mixing * densR_new.real + (1 - mixing) * densR

        history.append(densR.copy())

        # 5. Convergence check
        diff = np.sum(np.abs(densR - history[-2])) * dvol
        N_el = np.sum(np.abs(densR)) * dvol
        iter_time = time.perf_counter() - iter_start

        print(
            f"  KS iter {it:3d} | "
            f"dn = {diff:.6e} | "
            f"N_el = {N_el:.4f} | "
            f"{iter_time:.2f}s"
        )

        if convergence_threshold is not None and diff < convergence_threshold:
            print(f"  Converged at iteration {it} (dn < {convergence_threshold})")
            break

    total_time = time.perf_counter() - total_start
    print(f"KS loop completed in {total_time:.1f}s ({len(history) - 1} iterations)")

    return densR, history
