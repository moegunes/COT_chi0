"""
grid.py — Real-space and reciprocal-space grid construction
============================================================

Extracts the ``config()`` function from utils.py into a clean module.
Grid setup is deterministic and depends only on MaterialConfig parameters.

The main entry point is ``setup_grid(cfg)`` which returns a ``Grid`` object
containing all spatial grids, G-vectors, and Hartree prefactors.
"""

from dataclasses import dataclass

import numpy as np

from .config import MaterialConfig


@dataclass
class Grid:
    """Container for all spatial grid data.

    Attributes
    ----------
    rlist : np.ndarray
        Real-space grid coordinates, shape (N, 3).
    rlist_int : np.ndarray
        Integer grid indices, shape (N, 3).
    K : np.ndarray
        k-point vectors in Cartesian coords, shape (n_k, 3).
    K_int : np.ndarray
        k-point vectors in fractional coords, shape (n_k, 3).
    G_k_abinit_int : list of list
        G-vectors (integer) for each k-point [n_k][n_G_k][3].
    G_k_abinit : list of list
        G-vectors (Cartesian) for each k-point [n_k][n_G_k][3].
    G_dens : np.ndarray
        Density G-vectors (Cartesian), shape (n_G, 3).
    G_dens_int : np.ndarray
        Density G-vectors (integer), shape (n_G, 3).
    G_dic : dict
        Mapping from str([i,j,k]) -> index in G_dens.
    gnorm : np.ndarray
        |G| for each density G-vector, shape (n_G,).
    HP_nodens : np.ndarray
        Hartree prefactors 4π/|G|² (0 for G=0), shape (n_G,).
    Mt : np.ndarray
        Reciprocal lattice vectors matrix (3×3).
    n_rgrid : int
        Grid size per dimension.
    """

    rlist: np.ndarray
    MtR: np.ndarray
    R_list: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    rz: np.ndarray
    G_dens: np.ndarray
    G_dens_int: np.ndarray
    gnorm: np.ndarray
    HP_nodens: np.ndarray
    n_rgrid: int
    x_points: np.ndarray
    r_points: np.ndarray
    traj: np.ndarray


def setup_grid(cfg: MaterialConfig) -> Grid:
    """Construct all spatial grids from material configuration.

    This replaces the ``config()`` function from utils.py.
    Grid construction is vectorized where possible for speed.

    Parameters
    ----------
    cfg : MaterialConfig
        Material configuration from ``load_config()``.

    Returns
    -------
    Grid
        All grid data needed for subsequent calculations.
    """
    n = cfg.n_rgrid
    MtR = cfg.MtR
    Ecut = cfg.ecut
    L = cfg.a_l

    # ---- Real-space grid (vectorized) ----
    # Original: triple loop over z, y, x
    frac = np.arange(n) / n
    zz, yy, xx = np.meshgrid(frac, frac, frac, indexing="ij")
    frac_coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    rlist = frac_coords @ MtR.T  # shape (N, 3)

    # ---- Reciprocal lattice vectors ----
    Mt = 2 * np.pi * np.linalg.inv(MtR)

    # ---- Periodized R points for Lindhard function ----
    R_list = np.array(
        [
            a * MtR[0] + b * MtR[1] + c * MtR[2]
            for a in range(2)
            for b in range(2)
            for c in range(2)
        ]
    )
    rx = np.ascontiguousarray(rlist[:, 0])
    ry = np.ascontiguousarray(rlist[:, 1])
    rz = np.ascontiguousarray(rlist[:, 2])

    # ---- Density G-vectors ----
    qmax2 = n // 2
    di = np.arange(-qmax2, qmax2)
    dgi, dgj, dgk = np.meshgrid(di, di, di, indexing="ij")
    all_dg_int = np.column_stack([dgi.ravel(), dgj.ravel(), dgk.ravel()])
    all_dg_cart = all_dg_int @ Mt.T

    # Filter: |G|²/2 < 4*Ecut
    energy_dens = np.sum(all_dg_cart**2, axis=1) / 2.0
    mask_dens = energy_dens < 4 * Ecut

    G_dens = all_dg_cart[mask_dens]
    G_dens_int_arr = all_dg_int[mask_dens]

    # Build G_dic mapping
    G_dic = {}
    for idx, g in enumerate(G_dens_int_arr):
        G_dic[str(g.tolist())] = idx

    gnorm = np.linalg.norm(G_dens, axis=1)

    # ---- Hartree prefactors ----
    # 4π/|G|² for G≠0, 0 for G=0
    gnorm_safe = np.where(gnorm == 0, 1.0, gnorm)  # avoid division by zero
    HP_nodens = np.where(gnorm == 0, 0.0, 4 * np.pi / gnorm_safe**2)

    x_points = get_x_points(n, L)
    r_points = get_r_points(n, L)
    traj = get_traj(n)
    print("Grid setup complete.")

    print(
        f"Kohn-Sham potential loaded from ABINIT.\nGrid: {n}x{n}x{n},\nEcut: {Ecut} Hartree with {len(gnorm)} plane waves."
    )

    return Grid(
        rlist=np.array(rlist),
        MtR=MtR,
        R_list=R_list,
        rx=rx,
        ry=ry,
        rz=rz,
        G_dens=G_dens,
        G_dens_int=G_dens_int_arr,
        gnorm=gnorm,
        HP_nodens=HP_nodens,
        n_rgrid=n,
        x_points=x_points,
        r_points=r_points,
        traj=traj,
    )


def get_traj(n_rgrid: int) -> np.ndarray:
    """Get indices for the 001-110-111 diagonal trajectory through the cell.

    This is used for 1D plots along high-symmetry directions.

    Parameters
    ----------
    n_rgrid : int
        Grid size per dimension.

    Returns
    -------
    np.ndarray
        1D array of flat indices into the (n_rgrid³,) arrays.
    """
    traj = []
    n = n_rgrid

    # 001 direction
    for i in range(n + 1):
        ii, jj, kk = 0, 0, i % n
        traj.append(kk + jj * n + ii * n**2)

    # 110 direction
    for i in range(1, n):
        ii = i % n
        jj = i % n
        kk = 0
        traj.append(kk + jj * n + ii * n**2)

    # 111 direction
    for i in range(n):
        ii = i % n
        jj = ii
        kk = ii
        traj.append(kk + jj * n + ii * n**2)

    return np.array(traj)


def get_x_points(n_rgrid, L):
    return np.linspace(0, L, n_rgrid, endpoint=False)


def get_r_points(n_rgrid, L):
    return np.concatenate(
        [
            np.linspace(0, L, n_rgrid, endpoint=False),
            np.linspace(L, L + np.sqrt(2) * L, n_rgrid, endpoint=False),
            np.linspace(
                L + np.sqrt(2) * L,
                L + np.sqrt(2) * L + np.sqrt(3) * L,
                n_rgrid,
                endpoint=False,
            ),
        ]
    )
