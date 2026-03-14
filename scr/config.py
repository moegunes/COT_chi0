"""
config.py — Material configuration for cubic Helium
====================================================

Replaces the three input_inv*.py files with a single, explicit loader.
No global side effects: everything is returned in a dictionary.

Usage
-----
    cfg = load_config(ecut=15, a_l=8.016, data_dir="./data/equilibrium")
    # cfg is a dict with all material/grid parameters
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MaterialConfig:
    """All material and grid parameters for a cubic He calculation.

    Attributes
    ----------
    ecut : float
        Plane-wave energy cutoff in Hartree (e.g. 15 or 150).
    a_l : float
        Lattice constant in Bohr (equilibrium: 8.016).
    n_occup : int
        Number of occupied bands (1 for He).
    MtR : np.ndarray
        Real-space lattice vectors matrix (3×3).
    V_unitcell : float
        Unit cell volume in Bohr³.
    kx : np.ndarray
        k-point sampling array.
    V_ksR : np.ndarray
        Kohn-Sham potential in real space, loaded from file.
    V_extR : np.ndarray
        External (local pseudopotential) potential in real space.
    densR_ref : np.ndarray
        Reference charge density in real space, loaded from file.
    n_rgrid : int
        Grid size per dimension (cube root of len(V_ksR)).
    data_dir : str
        Path to the data directory used.
    """

    ecut: float
    a_l: float
    MtR: np.ndarray
    V_ksR: np.ndarray
    V_extR: np.ndarray
    densR_ref: np.ndarray
    n_rgrid: int
    data_dir: str

    # Allow numpy arrays in dataclass
    class Config:
        arbitrary_types_allowed = True


def load_config(
    ecut: int = 15,
    a_l: float = 8.016,
    data_dir: str = "./data/equilibrium",
    vks_file: Optional[str] = None,
    dens_file: Optional[str] = None,
    vext_file: Optional[str] = None,
) -> MaterialConfig:
    """Load material configuration for cubic Helium.

    Parameters
    ----------
    ecut : int
        Plane-wave cutoff energy in Hartree. Determines which data files
        to load (VksR_e{ecut}.dat, density_e{ecut}.dat).
    a_l : float
        Lattice constant in Bohr. Default 8.016 (equilibrium fcc He).
    data_dir : str
        Directory containing the .dat files.
    vks_file : str, optional
        Override filename for the Kohn-Sham potential. If None, uses
        ``VksR_e{ecut}.dat``.
    dens_file : str, optional
        Override filename for the reference density. If None, uses
        ``density_e{ecut}.dat``.
    n_occup : int
        Number of occupied bands (default 1 for He).
    kx : np.ndarray, optional
        k-point grid. Default: ``np.arange(-1, 2) * 1/3``.

    Returns
    -------
    MaterialConfig
        Dataclass with all parameters needed for downstream calculations.

    Examples
    --------
    >>> cfg = load_config(ecut=15, a_l=8.016)
    >>> print(cfg.n_rgrid, cfg.V_unitcell)
    """
    data_path = Path(data_dir)

    MtR = np.identity(3) * a_l
    V_unitcell = a_l**3  # simple cubic

    # --- Load Kohn-Sham potential ---
    if vks_file is None:
        vks_file = f"V_ksR_e{ecut}.dat"
    vks_path = data_path / vks_file
    V_ksR = np.genfromtxt(str(vks_path))

    # --- Load external (local pseudopotential) potential ---
    if vext_file is None:
        vext_file = f"VPS_loc_e{ecut}.dat"
    vext_path = data_path / vext_file
    V_extR = np.genfromtxt(str(vext_path))

    # --- Load reference density ---
    if dens_file is None:
        dens_file = f"density_e{ecut}.dat"
    dens_path = data_path / dens_file
    densR_ref = np.genfromtxt(str(dens_path))

    # --- Grid dimensions ---
    n_rgrid = int(np.round(len(V_ksR) ** (1.0 / 3.0)))

    return MaterialConfig(
        ecut=ecut,
        a_l=a_l,
        MtR=MtR,
        V_ksR=V_ksR,
        V_extR=V_extR,
        densR_ref=densR_ref,
        n_rgrid=n_rgrid,
        data_dir=str(data_dir),
    )
