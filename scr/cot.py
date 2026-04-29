import multiprocessing as mp
import time

import numpy as np

from .config import MaterialConfig
from .grid import Grid
from .heg import chiReal, chiReal_numba, n_h
from .icot import COT0, COT1, COT1_ALPHA, COT1_AV, COT1_AV_KS, COT1_KS
from .ks_loop_numba import run_ks_loop_fast
from .utils import vc_solutions_numba

try:
    import numba

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _compute_COT1_numba(V_ksR, rx, ry, rz, R_list, indices):
        """Numba-compiled parallel COT1 density computation."""

        n_pts = len(indices)
        N = len(V_ksR)
        results = np.empty(n_pts)
        pi = np.pi

        for idx in numba.prange(n_pts):
            i = indices[idx]
            v0 = V_ksR[i]

            chi_V_sum = 0.0
            chi_sum = 0.0
            for j in range(N):
                chi_val = chiReal_numba(v0, i, j, rx, ry, rz, R_list)
                chi_V_sum += chi_val * V_ksR[j]
                chi_sum += chi_val

            Vcon = chi_V_sum / chi_sum
            kF_c = np.sqrt(-2.0 * Vcon)
            results[idx] = kF_c * kF_c * kF_c / (3.0 * pi * pi)

        return results

    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _compute_COT1_av_numba(approximation, V_ksR, rx, ry, rz, R_list, indices):
        """Numba-compiled parallel COT1-av / COT1-alpha density computation."""
        n_pts = len(indices)
        N = len(V_ksR)
        n_R = len(R_list)
        results = np.empty(n_pts)
        pi = np.pi
        vcl = np.empty(n_pts)
        dvol = (rx[1] - rx[0]) ** 3
        if approximation == COT1_ALPHA:
            n_lpa = (-2 * V_ksR) ** (3 / 2) / (3 * pi**2)
            A = 0.7165
            B = 0.1919
            alpha = A * n_lpa**B

        for idx in numba.prange(n_pts):
            i = indices[idx]
            if approximation == COT1_AV:
                v0_chi = (V_ksR[i] + V_ksR) / 2.0
            elif approximation == COT1_ALPHA:
                v0_chi = V_ksR[i] / 2.0 + alpha[i] * V_ksR
            else:
                print("Invalid approximation type in _compute_COT1_av_numba")
            #    return [np.nan for _ in indices]
            chi_V_sum = 0.0
            chi_sum = 0.0
            """
            for j in range(N):
                chi_val = chiReal_numba(v0_chi[j], i, j, rx, ry, rz, R_list)
                chi_V_sum += chi_val * V_ksR[j]
                chi_sum += chi_val
            """
            kF = np.sqrt(-2.0 * (v0_chi - 1e-10))
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

            for k in range(n_R):
                Rx = R_list[k, 0]
                Ry = R_list[k, 1]
                Rz = R_list[k, 2]
                for j in range(N):
                    ddx = vx[j] - Rx
                    ddy = vy[j] - Ry
                    ddz = vz[j] - Rz
                    r = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                    u = kF2[j] * r
                    denom = kF2[j] * (r + 1e-15)
                    denom2 = denom * denom
                    denom4 = denom2 * denom2

                    chi_val = -factor[j] * (np.sin(u) - u * np.cos(u)) / denom4
                    chi_V_sum += chi_val * V_ksR[j]
                    chi_sum += chi_val

            # Fallback to COT1 connector when |v0| is too small for cubic solver
            # if abs(V_ksR[i]) < 1e-5:
            #    Vcon = chi_V_sum / chi_sum
            # else:
            Vcon = vc_solutions_numba(V_ksR[i], chi_V_sum * dvol)
            # if Vcon > 0:
            # print("Warning: Vcon=is small and positive. Shifting to negative...")
            #    print(Vcon, i)
            #    Vcon = -1e-6
            kF_c = np.sqrt(-2.0 * Vcon)
            vcl[idx] = Vcon
            results[idx] = kF_c * kF_c * kF_c / (3.0 * pi * pi)
        if vcl.max() > 0:
            #    print("Warning: vc has positive values, shifting to be negative...")
            for idx in numba.prange(n_pts):
                Vcon = vcl[idx] - vcl.max()
                kF_c = np.sqrt(-2.0 * Vcon)
                results[idx] = kF_c * kF_c * kF_c / (3.0 * pi * pi)
        return results

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

# Module-level worker state for shared-memory multiprocessing
_w = {}


def _init_COT1_worker(V_ksR, rx, ry, rz, R_list):
    """Called once per worker to attach shared data (no per-task pickling)."""
    global _w
    _w["V_ksR"] = V_ksR
    _w["rx"] = rx
    _w["ry"] = ry
    _w["rz"] = rz
    _w["R_list"] = R_list


def _compute_COT1_point(i):
    """Compute COT1 density at a single grid point using shared worker data."""
    V_ksR = _w["V_ksR"]
    rx, ry, rz = _w["rx"], _w["ry"], _w["rz"]
    R_list = _w["R_list"]

    v0 = V_ksR[i]
    kF = np.sqrt(-2 * (v0 - 1e-10))
    NF = kF / (2 * np.pi**2)
    n = kF**3 / (6 * np.pi**2)
    factor = 12 * np.pi * n * NF * 2
    kF2 = 2 * kF

    vx = np.abs(rx - rx[i])
    vy = np.abs(ry - ry[i])
    vz = np.abs(rz - rz[i])

    chi_V_sum = 0.0
    chi_sum = 0.0

    for R in R_list:
        r = np.sqrt((vx - R[0]) ** 2 + (vy - R[1]) ** 2 + (vz - R[2]) ** 2)
        u = kF2 * r
        denom = (kF2 * (r + 1e-15)) ** 4
        chi = -factor * (np.sin(u) - u * np.cos(u)) / denom

        chi_V_sum += np.dot(chi, V_ksR)
        chi_sum += chi.sum()

    Vcon = chi_V_sum / chi_sum
    kF_c = np.sqrt(-2 * Vcon)
    return kF_c**3 / (3 * np.pi**2)


def get_dens_parallel(
    system: MaterialConfig,
    grid: Grid,
    approximation: int,
    full_grid: bool = False,
):
    """COT1 density via Numba (preferred) or shared-memory multiprocessing."""

    rlist = grid.rlist
    N = len(rlist)
    R_list = grid.R_list
    rx = grid.rx
    ry = grid.ry
    rz = grid.rz

    if system.V_ksR.max() > 0:
        print("Warning: V_ksR has positive values, shifting to be negative...")
        system.V_ksR = system.V_ksR - system.V_ksR.max() - 1e-6

    V_ksR = system.V_ksR

    if approximation == COT0:
        return n_h(system.V_ksR)

    if full_grid:
        indices = np.arange(N, dtype=np.int64)
    else:
        indices = np.asarray(grid.traj, dtype=np.int64)

    args = (np.asarray(V_ksR), rx, ry, rz, R_list, indices)

    n_pts = len(indices)

    if _HAS_NUMBA:
        # n_threads = numba.config.NUMBA_NUM_THREADS
        n_threads = numba.get_num_threads()
        print(f"Processing {n_pts} grid points with Numba ({n_threads} threads)...")
        start_time = time.perf_counter()
        if approximation == COT1:
            density_vals = _compute_COT1_numba(*args)

        elif approximation in (COT1_AV, COT1_ALPHA):
            density_vals = _compute_COT1_av_numba(approximation, *args)

        elif approximation in (COT1_AV_KS, COT1_KS):
            shift_density = False
            density_vals, history = run_ks_loop_fast(
                system, grid, approximation, shift_density=shift_density
            )

        results = np.zeros(N)
        results[indices] = density_vals
        end_time = time.perf_counter()
        elapsed = end_time - start_time

        print(f"Numba computation finished in {elapsed:.3f} seconds")
        return results

    # Fallback: shared-memory multiprocessing
    from tqdm import tqdm

    n_workers = mp.cpu_count()
    print(f"Processing {n_pts} grid points using {n_workers} workers (shared data)...")

    with mp.Pool(
        n_workers,
        initializer=_init_COT1_worker,
        initargs=(V_ksR, rx, ry, rz, R_list),
    ) as pool:
        density_list = list(
            tqdm(
                pool.imap(_compute_COT1_point, indices, chunksize=100),
                total=n_pts,
                desc="Computing density",
            )
        )

    results = np.zeros(N)
    results[indices] = np.array(density_list)

    print("=" * 42)
    print("SUCCESSFULLY COMPLETED.")
    print("=" * 42)
    return results


def process(
    system: MaterialConfig,
    grid: Grid,
    approximation: str,
    i: int,
    mu: float = 0.0,
    interacting: bool = False,
):
    """Modern, efficient replacement for the old `process` function.

    Relies on module-level globals used throughout the project (e.g. `rvec`,
    `vbar`, `V_ksR`, `n_rgrid1`, `V_unitcell`, `n_h`, `chiReal`, `gauss_MIC`).
    The function uses Python 3.10 `match` for clarity and factors common work
    into small local helpers.
    """
    V_ksR = system.V_ksR

    dx = system.a_l / system.n_rgrid
    vol = dx**3

    def _compute_Vcon(v0_chi):
        chiR = chiReal(v0_chi, i, grid)
        integrand = chiR * V_ksR
        chiq0 = np.sum(chiR)
        Vcon = np.sum(integrand) / chiq0
        return Vcon

    match approximation:
        case "COT0":
            Vcon = V_ksR[i]
            return n_h(Vcon)

        case "lra":
            v0 = V_ksR[i]
            v0_chi = v0
            chiR = chiReal(v0_chi, i, grid)
            integrand = chiR * (V_ksR - v0)
            first_order = np.sum(integrand * vol).real
            return n_h(v0) + first_order

        case "COT1":
            v0 = V_ksR[i]
            v0_chi = v0
            Vcon = _compute_Vcon(v0_chi)
            return n_h(Vcon)

        case "COT1-av":
            from .utils import vc_solutions

            v0 = V_ksR[i]
            v0_chi = (v0 + V_ksR) / 2
            chiR = chiReal(v0_chi, i, grid)
            integrand = chiR * V_ksR
            numerator = np.sum(integrand * vol).real
            vc = vc_solutions(v0, numerator)
            return n_h(vc)

        case "COT1-alpha":
            from .utils import vc_solutions

            v0 = V_ksR[i]
            v0_chi = (v0 + V_ksR) / 2
            chiR = chiReal(v0_chi, i, grid)
            integrand = chiR * V_ksR
            numerator = np.sum(integrand * vol).real
            vc = vc_solutions(v0, numerator)
            return n_h(vc)

        case _:
            raise ValueError(f"Unknown choice '{approximation}' passed to process()")


def get_dens_ks_loop(
    system: MaterialConfig,
    grid: Grid,
    cot_approximation: int = COT1_AV,
    n_iter: int = 100,
    mixing: float = 0.65,
    n_electrons: float = 2,
    densR_init=None,
    convergence_threshold=None,
    full_grid: bool = True,
    shift_density: bool = False,
):
    """Run a Kohn-Sham self-consistent loop using a COT density functional.

    Instead of diagonalizing the KS Hamiltonian, computes the density
    from V_KS using a COT approximation at each self-consistent step.

    Parameters
    ----------
    system : MaterialConfig
        Material configuration (must include V_extR).
    grid : Grid
        Spatial grid data.
    cot_approximation : int
        COT approximation for the density step (COT0, COT1, COT1_AV, etc.).
    n_iter : int
        Maximum number of self-consistent iterations.
    mixing : float
        Linear mixing: n = mixing * n_COT + (1-mixing) * n_old.
    n_electrons : float
        Target number of electrons (2 for He).
    densR_init : np.ndarray, optional
        Initial density. If None, uses uniform density.
    convergence_threshold : float, optional
        Stop when integral |n_new - n_old| dr < threshold.
    full_grid : bool
        Compute density on full grid (True) or trajectory only (False).
    shift_density : bool
        If True, shift density for V_xc to conserve electron number
        (paper Sec. V modified SC-COT). Default False.

    Returns
    -------
    densR : np.ndarray
        Final converged density.
    history : list of np.ndarray
        Density at each iteration.
    """
    from .ks_loop import run_ks_loop

    return run_ks_loop(
        system=system,
        grid=grid,
        approximation=cot_approximation,
        n_iter=n_iter,
        mixing=mixing,
        n_electrons=2,
        densR_init=densR_init,
        convergence_threshold=convergence_threshold,
        full_grid=full_grid,
        shift_density=shift_density,
    )


# ---------------------------------------------------------------------------
# Numba-accelerated KS self-consistent loop
# ---------------------------------------------------------------------------
# TODO: once validated, merge get_dens_ks_loop_fast into get_dens_parallel
#       by adding  COT1_AV_KS = 4  handling inside that function.


def get_dens_ks_loop_fast(
    system: MaterialConfig,
    grid: Grid,
    cot_approximation: int = COT1_AV,
    n_iter: int = 100,
    mixing: float = 0.65,
    n_electrons: float = 2,
    densR_init=None,
    convergence_threshold=None,
    full_grid: bool = True,
    shift_density: bool = False,
):
    """Numba-accelerated KS self-consistent loop (drop-in for get_dens_ks_loop).

    Uses ``ks_loop_numba.run_ks_loop_fast`` which pre-computes ``V_ext_G``
    once and employs numba-compiled LDA, density normalisation, and mixing
    kernels.  The inner O(N²) density step calls the same
    ``_compute_COT1_av_numba`` / ``_compute_COT1_numba`` kernels as
    ``get_dens_parallel``, so results are numerically identical to
    ``get_dens_ks_loop``.

    Parameters
    ----------
    system : MaterialConfig
    grid : Grid
    cot_approximation : int
        COT approximation (COT0=0, COT1=1, COT1_AV=2, COT1_ALPHA=3).
    n_iter : int
    mixing : float
    n_electrons : float
    densR_init : np.ndarray, optional
    convergence_threshold : float, optional
    full_grid : bool
    shift_density : bool

    Returns
    -------
    densR : np.ndarray
    history : list of np.ndarray
    """
    from .ks_loop_numba import run_ks_loop_fast

    return run_ks_loop_fast(
        system=system,
        grid=grid,
        approximation=cot_approximation,
        n_iter=n_iter,
        mixing=mixing,
        n_electrons=2,
        densR_init=densR_init,
        convergence_threshold=convergence_threshold,
        shift_density=shift_density,
    )
