import numba
import numpy as np


def cubic_pos(spaces):
    ndim = len(spaces)
    gvecs = np.stack(np.meshgrid(*spaces, indexing="ij"), axis=-1).reshape(-1, ndim)
    return gvecs


def get_gvecs(mesh):
    spaces = [np.arange(nx) for nx in mesh]
    return cubic_pos(spaces)


def get_rvecs(axes, mesh, center=False):
    gvecs = get_gvecs(mesh)
    fracs = axes / np.array(mesh)[:, np.newaxis]  # axes is row-major
    rvecs = np.dot(gvecs, fracs)
    if center:
        c = 0.5 * np.ones(len(mesh)) @ (axes / np.array(mesh))
        rvecs += c
    return rvecs


def get_G(cut, axes, mesh):
    """
    Get G-vectors for a given energy cutoff and grid size.
    """
    Ecut = cut
    n_rgrid1, n_rgrid2, n_rgrid3 = mesh[0], mesh[1], mesh[2]
    MtR = axes[[2, 1, 0]]
    Mt = 2 * np.pi * (np.linalg.inv(MtR))

    G_dens, G_dens_int = [], []
    qmax2_x = int(n_rgrid1 / 2)  # 2int(np.sqrt(8*Ecut)/np.linalg.norm(Mt[:,0]))*2
    qmax2_y = int(n_rgrid2 / 2)
    qmax2_z = int(n_rgrid3 / 2)
    G_dic = {}
    count = 0
    for i in range(-qmax2_x, qmax2_x):
        for j in range(-qmax2_y, qmax2_y):
            for k in range(-qmax2_z, qmax2_z):
                g_orth = np.dot(Mt, [i, j, k])
                if np.linalg.norm(g_orth) ** 2 / 2 < 4 * Ecut:
                    # print([i,j,k], '>>' , np.linalg.norm(q+[i,j,k])**2/2)
                    x = g_orth[0]  # round(g_orth[0],5)
                    y = g_orth[1]  # round(g_orth[1],5)
                    z = g_orth[2]  # round(g_orth[2],5)
                    G_dens.append([x, y, z])
                    G_dic[str([i, j, k])] = count
                    G_dens_int.append([i, j, k])
                    count += 1

    gnorm = np.linalg.norm(G_dens, axis=1)  ## useful for connector
    return gnorm, G_dens, G_dens_int


@numba.njit(cache=True, inline="always")
def vc_solutions_numba(v0, numerator):
    A0 = v0
    B0 = numerator * numerator * np.pi**4

    J = 1j
    sqrt3 = np.sqrt(3.0 + 0j)

    # Force complex arithmetic so np.sqrt handles negative discriminants
    inner_sqrt = np.sqrt((4.0 * A0**3 * B0 + 27.0 * B0**2) + 0j)
    D = -2.0 * A0**3 - 27.0 * B0 + 3.0 * sqrt3 * inner_sqrt

    D_cuberoot = D ** (1.0 / 3.0)

    two_1_3 = 2.0 ** (1.0 / 3.0)
    two_2_3 = 2.0 ** (2.0 / 3.0)

    # Solution 1
    vc1 = (1.0 / 3.0) * (-A0 + (two_1_3 * A0**2) / D_cuberoot + D_cuberoot / two_1_3)

    # complex constants
    w_plus = 1.0 + J * sqrt3
    w_minus = 1.0 - J * sqrt3

    # Solution 2
    vc2 = (
        -(A0 / 3.0)
        - (w_plus * A0**2) / (3.0 * two_2_3 * D_cuberoot)
        - (w_minus * D_cuberoot) / (6.0 * two_1_3)
    )

    # Solution 3
    vc3 = (
        -(A0 / 3.0)
        - (w_minus * A0**2) / (3.0 * two_2_3 * D_cuberoot)
        - (w_plus * D_cuberoot) / (6.0 * two_1_3)
    )

    # Return the solution with smallest |imag| (should be unique real root)
    a1 = np.abs(vc1.imag)
    a2 = np.abs(vc2.imag)
    a3 = np.abs(vc3.imag)

    if a1 <= a2 and a1 <= a3:
        return vc1.real
    elif a2 <= a3:
        return vc2.real
    else:
        return vc3.real
