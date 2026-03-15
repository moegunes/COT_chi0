import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 (registers styles on import)

from scr.config import load_config
from scr.cot import get_dens_parallel
from scr.grid import get_traj, setup_grid

"""
LRA = -1
COT0 = 0
COT1 = 1
COT1_AV = 2
COT1_ALPHA = 3
COT1_KS = 11
COT1_AV_KS = 22
"""


def main():
    L = 8.016
    Ecut = 15

    system = load_config(ecut=Ecut, a_l=L)
    grid = setup_grid(system)
    traj1 = get_traj(grid.n_rgrid)

    V_ksR = system.V_ksR  # noqa: F841

    approximation = 2

    density = get_dens_parallel(system, grid, approximation, full_grid=True)

    plt.style.use("science")

    densR_ref = system.densR_ref
    r_points = grid.r_points
    fig, ax = plt.subplots(dpi=200)
    ax.plot(r_points, densR_ref[traj1], "k", label="Reference")
    ax.plot(r_points, density[traj1], "r", label=f"{approximation}")
    ax.set_xlabel(r"$r$ (Bohr)")
    ax.set_ylabel(r"$n(r)$ (Bohr$^{-3}$)")
    ax.legend()
    ax.set_xlim(0, L)
    plt.tight_layout()
    plt.show()

    # np.savetxt(f"n_{approximation}_v3.dat", density)


if __name__ == "__main__":
    main()
