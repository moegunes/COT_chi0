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


def vc_solutions(v0, numerator):
    """
    Returns the three solutions for self-consistent vc.

    Notes:
    - Uses principal branches for complex sqrt and complex cube root (via **(1/3)).
    - A0, B0 can be real or complex (Python numbers).
    """
    import cmath

    J = 1j
    sqrt3 = cmath.sqrt(3)

    A0 = v0
    B0 = numerator**2 * np.pi**4

    inner_sqrt = cmath.sqrt(4 * A0**3 * B0 + 27 * B0**2)
    D = -2 * A0**3 - 27 * B0 + 3 * sqrt3 * inner_sqrt

    # Cube root and 2^(1/3), 2^(2/3)
    D_cuberoot = (
        D ** (1 / 3) if not isinstance(D, complex) else cmath.exp(cmath.log(D) / 3)
    )
    two_1_3 = 2 ** (1 / 3)
    two_2_3 = 2 ** (2 / 3)

    # Solution 1:
    vc1 = (1 / 3) * (-A0 + (two_1_3 * A0**2) / (D_cuberoot) + (D_cuberoot) / (two_1_3))

    # Common complex factors (1 ± i*sqrt(3))
    w_plus = 1 + J * sqrt3
    w_minus = 1 - J * sqrt3

    # Solution 2:
    vc2 = (
        -(A0 / 3)
        - (w_plus * A0**2) / (3 * two_2_3 * D_cuberoot)
        - (w_minus * D_cuberoot) / (6 * two_1_3)
    )

    # Solution 3:
    vc3 = (
        -(A0 / 3)
        - (w_minus * A0**2) / (3 * two_2_3 * D_cuberoot)
        - (w_plus * D_cuberoot) / (6 * two_1_3)
    )
    # we return only the real root. for this, we need to check which of the three solutions is real (or has the smallest imaginary part)
    solutions = [vc1, vc2, vc3]
    real_solutions = [s for s in solutions if abs(s.imag) < 1e-6]
    if real_solutions:
        if len(real_solutions) > 1:
            raise ValueError(
                "Multiple real solutions found when solving the self-consistent equation for the connector. Solutions are: \n"
                + ", \n".join(str(np.round(s, 4)) for s in real_solutions)
            )
        else:
            return real_solutions[0].real
    else:
        raise ValueError(
            "No real solution found for vc. Solutions are: \n"
            + ", \n".join(str(np.round(s, 4)) for s in solutions)
        )


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
