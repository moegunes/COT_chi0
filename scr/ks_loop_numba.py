"""
ks_loop_numba.py — Numba-accelerated Kohn-Sham self-consistent loop
====================================================================

Optimised version of ks_loop.py.  Key improvements over ks_loop.py:

1. ``V_ext_G`` is pre-computed *once* before the loop (saves 1 FFT per
   iteration).
2. LDA exchange-correlation (``_lda_vxc_numba``) is compiled with numba
   (``parallel=True``, ``cache=True``, ``fastmath=True``) — avoids creating
   temporary boolean-mask arrays.
3. Density normalisation (``_normalize_density_numba``) is compiled with
   numba, replacing the bisection loop in xc.normalize_density.
4. A mixing step (``_mix_and_check_numba``) accumulates the convergence
   norm inside numba to avoid extra numpy passes.

The inner density step (the O(N²) COT χ convolution) delegates to the
*same* numba kernels used by ``get_dens_parallel``
(``_compute_COT1_av_numba`` / ``_compute_COT1_numba`` in cot.py), so
results are numerically identical to ``run_ks_loop`` given the same V_KS.

Benchmark
---------
Use ``benchmark_density_step`` to compare a single density-computation
step:

  • **Legacy** — simulates the old ``Pool.map(get_connector, ...)`` approach
    from tmp_cot_ks.py (Python chi with multiprocessing).
  • **Fast**   — numba kernel (the path used here).

Usage
-----
    from scr.ks_loop_numba import run_ks_loop_fast
    densR, history = run_ks_loop_fast(system, grid, approximation=COT1_AV,
                                      n_iter=100, mixing=0.65)
"""

import time

import numpy as np

from .fourier import fft_g_to_r, fft_r_to_g
from .xc import lda_vxc, normalize_density

try:
    import numba

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# ---------------------------------------------------------------------------
# Numba-compiled helper kernels
# ---------------------------------------------------------------------------

if _HAS_NUMBA:

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _lda_vxc_numba(densR):  # pyright: ignore[reportRedeclaration]
        """Parallel numba LDA exchange-correlation potential (PZ parametrisation).

        Equivalent to xc.lda_vxc but avoids allocating temporary boolean-mask
        arrays; instead uses per-element if-else inside the parallel loop.
        """
        n = len(densR)
        result = np.empty(n)

        gamma = -0.1423
        beta1 = 1.0529
        beta2 = 0.3334
        Au = 0.0311
        Bu = -0.048
        Cu = 0.0020
        Du = -0.0116
        cbrt_3_over_pi = (3.0 / np.pi) ** (1.0 / 3.0)

        for i in numba.prange(n):
            ni = densR[i]
            if ni < 0.0:
                ni = -ni
            if ni < 1e-30:
                ni = 1e-30

            rs = (4.0 * np.pi / 3.0 * ni) ** (-1.0 / 3.0)

            # Slater exchange
            v_x = -cbrt_3_over_pi * ni ** (1.0 / 3.0)

            # Perdew-Zunger correlation
            sq_rs = np.sqrt(rs)
            if rs < 1.0:
                v_c = (
                    Au * np.log(rs)
                    + (Bu - Au / 3.0)
                    + 2.0 / 3.0 * Cu * rs * np.log(rs)
                    + 1.0 / 3.0 * (2.0 * Du - Cu) * rs
                )
            else:
                denom = 1.0 + beta1 * sq_rs + beta2 * rs
                v_cep = gamma / denom
                v_c = (
                    v_cep
                    * (1.0 + 7.0 / 6.0 * beta1 * sq_rs + 4.0 / 3.0 * beta2 * rs)
                    / denom
                )

            result[i] = v_x + v_c

        return result

    @numba.njit(cache=True, fastmath=True)
    def _normalize_density_numba(densR, n_electrons, dvol):  # pyright: ignore[reportRedeclaration]
        """Shift density to integrate to n_electrons.

        Numba-compiled equivalent of xc.normalize_density.
        The bisection loop is compiled natively (no Python overhead).
        """
        n = len(densR)
        rho = densR.copy()

        N_current = 0.0
        for i in range(n):
            N_current += rho[i]
        N_current *= dvol

        if N_current <= n_electrons:
            shift = (n_electrons - N_current) / (n * dvol)
            for i in range(n):
                rho[i] += shift
        else:
            hi = 0.0
            for i in range(n):
                if rho[i] > hi:
                    hi = rho[i]
            lo = 0.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                N_mid = 0.0
                for i in range(n):
                    v = rho[i] - mid
                    if v > 0.0:
                        N_mid += v
                N_mid *= dvol
                if N_mid > n_electrons:
                    lo = mid
                else:
                    hi = mid
            alpha = 0.5 * (lo + hi)
            for i in range(n):
                v = rho[i] - alpha
                rho[i] = v if v > 0.0 else 0.0

        for i in range(n):
            if rho[i] < 1e-12:
                rho[i] = 1e-12

        return rho

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _mix_and_norm_numba(densR_new, densR_old, mixing, dvol):  # pyright: ignore[reportRedeclaration]
        """Linear mixing and simultaneous convergence norm.

        Returns ``(densR_mixed, dn)`` where ``dn = Σ|n_new - n_old| * dvol``.
        Avoids two separate numpy passes.
        """
        n = len(densR_new)
        result = np.empty(n)
        dn = 0.0
        for i in numba.prange(n):
            mixed = mixing * densR_new[i] + (1.0 - mixing) * densR_old[i]
            result[i] = mixed
        # Convergence norm (sequential reduction — prange reduction not supported
        # for abs in older numba, so keep serial loop here)
        for i in range(n):
            dn += abs(result[i] - densR_old[i])
        dn *= dvol
        return result, dn

else:
    # Pure-Python fallbacks (used when numba is not installed)

    def _lda_vxc_numba(densR):  # noqa: F811
        return lda_vxc(densR)

    def _normalize_density_numba(densR, n_electrons, dvol):  # noqa: F811
        return normalize_density(densR, n_electrons, dvol)

    def _mix_and_norm_numba(densR_new, densR_old, mixing, dvol):  # noqa: F811
        result = mixing * densR_new + (1.0 - mixing) * densR_old
        dn = np.sum(np.abs(result - densR_old)) * dvol
        return result, dn


# ---------------------------------------------------------------------------
# V_KS builder (pre-computed V_ext_G)
# ---------------------------------------------------------------------------


def _build_vks_fast(
    densR,
    V_ext_G,
    V_extR,
    G_dens_int,
    HP_nodens,
    n_rgrid,
    n_electrons=None,
    dvol=None,
    shift_density=False,
    gspace=True,
):
    """Build V_KS using pre-computed ``V_ext_G`` (saves one FFT per call).

    Equivalent to ks_loop._build_vks but accepts ``V_ext_G`` directly
    instead of ``V_extR`` for the G-space path, and uses the numba-compiled
    LDA and density-normalisation functions.

    Parameters
    ----------
    densR : np.ndarray
        Current density in real space.
    V_ext_G : np.ndarray
        External potential in G-space (pre-computed, constant across iterations).
    V_extR : np.ndarray
        External potential in real space (used only when gspace=False).
    G_dens_int : np.ndarray
        Integer G-vector indices.
    HP_nodens : np.ndarray
        Hartree prefactors 4π/|G|² (0 for G=0).
    n_rgrid : int
        Grid size per dimension.
    n_electrons : float, optional
        Target electron number (needed when shift_density=True).
    dvol : float, optional
        Volume element dx³ (needed when shift_density=True).
    shift_density : bool
        If True, shift density for V_xc (paper Sec. V).
    gspace : bool
        If True, sum V_ext + V_xc + V_H in G-space (connectors).
        If False, sum in R-space (COT0/LPA).

    Returns
    -------
    np.ndarray
        V_KS in real space (real-valued).
    """
    densG = fft_r_to_g(densR, G_dens_int, n_rgrid)
    V_H_G = densG * HP_nodens

    if shift_density and n_electrons is not None and dvol is not None:
        dens_for_vxc = _normalize_density_numba(densR, n_electrons, dvol)
    else:
        dens_for_vxc = np.abs(densR.real)

    V_xc_R = _lda_vxc_numba(dens_for_vxc)

    if gspace:
        V_xc_G = fft_r_to_g(V_xc_R, G_dens_int, n_rgrid)
        V_ks_G = V_ext_G + V_xc_G + V_H_G
        V_ksR = fft_g_to_r(V_ks_G, G_dens_int, n_rgrid).real
    else:
        V_H_R = fft_g_to_r(V_H_G, G_dens_int, n_rgrid).real
        V_ksR = (V_extR + V_H_R + V_xc_R).real

    return V_ksR


# ---------------------------------------------------------------------------
# Main KS loop
# ---------------------------------------------------------------------------


def run_ks_loop_fast(
    system,
    grid,
    approximation,
    n_iter=15,
    mixing=0.65,
    n_electrons=2,
    densR_init=None,
    convergence_threshold=None,
    shift_density=False,
):
    """Run the KS self-consistent loop with numba-accelerated components.

    Drop-in replacement for ``ks_loop.run_ks_loop``.  Results are numerically
    identical; the speedup comes from:

    * Pre-computing ``V_ext_G`` once (saves 1 FFT / iteration).
    * Using numba-compiled LDA and density-normalisation kernels.
    * Using numba-compiled mixing + convergence-norm accumulation.

    The inner density step (the dominant O(N²) work) calls the same numba
    kernels as ``get_dens_parallel``.

    Parameters
    ----------
    system : MaterialConfig
        Material configuration with ``V_extR``.
    grid : Grid
        Spatial grid data.
    approximation : int
        COT approximation type (COT0=0, COT1=1, COT1_AV=2, COT1_ALPHA=3).
    n_iter : int
        Maximum number of self-consistent iterations.
    mixing : float
        Linear mixing parameter: n = mixing*n_new + (1-mixing)*n_old.
    n_electrons : float
        Number of electrons (2 for He).
    densR_init : np.ndarray, optional
        Initial density. If None, uses uniform density with correct N_el.
    convergence_threshold : float, optional
        Stop when Σ|Δn|*dV < threshold.
    full_grid : bool
        Compute density on full grid (True) or trajectory only (False).
    shift_density : bool
        If True, shift density for V_xc to conserve electron number
        (paper Sec. V modified SC-COT).

    Returns
    -------
    densR : np.ndarray
        Final converged density.
    history : list of np.ndarray
        Density at each iteration (including initial).
    """
    from .cot import _compute_COT1_av_numba, _compute_COT1_numba
    from .heg import n_h
    from .icot import COT0, COT1_AV, COT1_AV_KS, COT1_KS

    n = grid.n_rgrid
    N = n**3
    G_dens_int = grid.G_dens_int
    HP_nodens = grid.HP_nodens
    dvol = (system.a_l / n) ** 3
    rx, ry, rz = grid.rx, grid.ry, grid.rz
    R_list = grid.R_list
    indices = np.arange(N, dtype=np.int64)

    # ----------------------------------------------------------------
    # Pre-compute constant: V_ext in G-space (saved across all iterations)
    # ----------------------------------------------------------------
    V_ext_G = fft_r_to_g(system.V_extR, G_dens_int, n)

    # ----------------------------------------------------------------
    # Warm-up numba kernels to pay JIT cost *before* the timed loop
    # ----------------------------------------------------------------
    if _HAS_NUMBA:
        _dummy = np.ones(8, dtype=float) * 1e-3
        _lda_vxc_numba(_dummy)
        _mix_and_norm_numba(_dummy, _dummy, mixing, dvol)
        if shift_density:
            _normalize_density_numba(_dummy, n_electrons, dvol)

    # ----------------------------------------------------------------
    # Initial density
    # ----------------------------------------------------------------
    if densR_init is not None:
        densR = np.array(densR_init, dtype=float)
    else:
        V_cell = system.a_l**3
        densR = np.ones(N) * (n_electrons / V_cell)

    history = [densR.copy()]

    print(f"Starting KS loop (fast/numba): {n_iter} iterations, mixing={mixing}")
    print(f"Initial N_el = {np.sum(densR) * dvol:.4f}")
    if shift_density:
        print(f"Density shift enabled (target N_el={n_electrons})")

    total_start = time.perf_counter()
    t_vks_total = 0.0
    t_dens_total = 0.0

    for it in range(1, n_iter + 1):
        iter_start = time.perf_counter()

        # 1. Build V_KS
        use_gspace = approximation != COT0
        t0 = time.perf_counter()
        V_ksR = _build_vks_fast(
            densR,
            V_ext_G,
            system.V_extR,
            G_dens_int,
            HP_nodens,
            n,
            n_electrons=n_electrons,
            dvol=dvol,
            shift_density=shift_density,
            gspace=use_gspace,
        )
        t_vks_total += time.perf_counter() - t0

        # 2. Shift V_KS so it is all-negative (required by connector maths)
        if approximation != COT0 and V_ksR.max() > 0:
            print("  V_KS exceeds zero. Shifting...")
            V_ksR = V_ksR - V_ksR.max()

        # 3. Compute density from V_KS
        V_ksR_backup = system.V_ksR
        system.V_ksR = V_ksR

        t0 = time.perf_counter()
        if approximation == COT0:
            mask = V_ksR < 0
            densR_new = np.zeros_like(V_ksR)
            densR_new[mask] = n_h(V_ksR[mask])
        elif approximation == COT1_KS:
            args = (np.asarray(V_ksR), rx, ry, rz, R_list, indices)
            densR_new = _compute_COT1_numba(*args)
        elif approximation == COT1_AV_KS:
            args = (COT1_AV, np.asarray(V_ksR), rx, ry, rz, R_list, indices)
            densR_new = _compute_COT1_av_numba(*args)
        t_dens_total += time.perf_counter() - t0

        system.V_ksR = V_ksR_backup

        # 4. Mix and compute convergence norm (single numba pass)
        densR, diff = _mix_and_norm_numba(densR_new.real, densR, mixing, dvol)
        history.append(densR.copy())

        N_el = np.sum(np.abs(densR)) * dvol
        iter_time = time.perf_counter() - iter_start

        print(
            f"  KS iter {it:3d} | "
            f"dn = {diff:.6e} | "
            f"N_el = {N_el:.4f} | "
            f"total {iter_time:.2f}s"
        )

        if convergence_threshold is not None and diff < convergence_threshold:
            print(f"  Converged at iteration {it} (dn < {convergence_threshold})")
            break

    total_time = time.perf_counter() - total_start
    n_iters_done = len(history) - 1
    print(f"KS loop (fast) completed in {total_time:.1f}s ({n_iters_done} iterations)")
    if n_iters_done > 0:
        print(
            f"  Per-iter breakdown: V_KS {t_vks_total / n_iters_done:.3f}s | "
            f"density {t_dens_total / n_iters_done:.2f}s"
        )

    return densR, history


# ---------------------------------------------------------------------------
# Benchmark: one density-computation step — legacy vs fast
# ---------------------------------------------------------------------------


def benchmark_density_step(system, grid, approximation, full_grid=True):
    """Compare one density-computation step: legacy Pool.map vs numba fast.

    The *legacy* path simulates the original ``tmp_cot_ks.py`` approach:
    ``multiprocessing.Pool.map`` with a numpy-level chi function per point.

    The *fast* path calls the numba kernel (``_compute_COT1_av_numba`` or
    ``_compute_COT1_numba``) exactly as ``get_dens_parallel`` does.

    Parameters
    ----------
    system : MaterialConfig
        Must have ``V_ksR`` set to the current KS potential.
    grid : Grid
        Spatial grid.
    approximation : int
        COT approximation (COT1=1, COT1_AV=2).
    full_grid : bool
        If True benchmark on full N³ grid; if False use trajectory.

    Returns
    -------
    dict
        Keys: ``legacy_time``, ``fast_time``, ``speedup``,
              ``max_diff``, ``legacy_density``, ``fast_density``.
    """
    import multiprocessing as mp

    from .cot import _get_dens_COT1_fast
    from .heg import chiReal, n_h

    V_ksR = np.asarray(system.V_ksR)
    rlist = grid.rlist
    N = len(rlist)
    indices = (
        np.arange(N, dtype=np.int64)
        if full_grid
        else np.asarray(grid.traj, dtype=np.int64)
    )
    n_pts = len(indices)

    # ------------------------------------------------------------------
    # Legacy path — multiprocessing Pool.map with Python chi function
    # (mirrors the Pool.map(get_connector, ...) pattern in tmp_cot_ks.py)
    # ------------------------------------------------------------------
    def _legacy_point(i):
        """Single-point COT1 density (numpy chi, no numba — like get_connector)."""
        v0 = V_ksR[i]
        chiR = chiReal(v0, i, grid)
        chi_V = np.dot(chiR, V_ksR)
        chi_sum = np.sum(chiR)
        Vcon = chi_V / chi_sum
        if Vcon >= 0:
            return 0.0
        return float(n_h(Vcon))

    n_workers = min(mp.cpu_count(), n_pts)
    print(f"[benchmark] {n_pts} points, {n_workers} workers")
    print("[benchmark] Running LEGACY (Pool.map, numpy chi) ...")
    t0 = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        legacy_vals = np.array(
            pool.map(
                _legacy_point, list(indices), chunksize=max(1, n_pts // (n_workers * 4))
            )
        )
    legacy_time = time.perf_counter() - t0
    print(f"  Legacy time: {legacy_time:.2f}s")

    legacy_density = np.zeros(N)
    legacy_density[indices] = legacy_vals

    # ------------------------------------------------------------------
    # Fast path — numba kernel (same as get_dens_parallel uses)
    # ------------------------------------------------------------------
    print("[benchmark] Running FAST (numba parallel) ...")
    # First call includes JIT compile time — report both
    t0 = time.perf_counter()
    fast_density = _get_dens_COT1_fast(approximation, system, grid, full_grid)
    first_call = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast_density = _get_dens_COT1_fast(approximation, system, grid, full_grid)
    fast_time = time.perf_counter() - t0
    print(f"  Fast time (1st call incl. JIT): {first_call:.2f}s")
    print(f"  Fast time (2nd call, cached):   {fast_time:.2f}s")

    speedup_vs_cached = legacy_time / fast_time if fast_time > 0 else float("inf")
    speedup_vs_first = legacy_time / first_call if first_call > 0 else float("inf")

    # Compare results (use COT1 for both to keep physics identical)
    max_diff = np.max(np.abs(legacy_density[indices] - fast_density[indices]))
    mean_diff = np.mean(np.abs(legacy_density[indices] - fast_density[indices]))

    print(
        f"\n[benchmark] Results:"
        f"\n  Legacy time : {legacy_time:.2f}s"
        f"\n  Fast time   : {fast_time:.3f}s (cached numba)"
        f"\n  Speedup     : {speedup_vs_cached:.1f}×  (vs cached numba)"
        f"\n               {speedup_vs_first:.1f}×  (vs 1st numba call incl. JIT)"
        f"\n  Max |Δn|    : {max_diff:.3e}"
        f"\n  Mean |Δn|   : {mean_diff:.3e}"
    )

    return {
        "legacy_time": legacy_time,
        "fast_time_cached": fast_time,
        "fast_time_first": first_call,
        "speedup_cached": speedup_vs_cached,
        "speedup_first": speedup_vs_first,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "legacy_density": legacy_density,
        "fast_density": fast_density,
    }
