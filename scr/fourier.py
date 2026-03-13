import sys

import numpy as np


def myifft_dens_v(
    fR, G_dens_int, rlist, idx=None
):  # inverse FT of the density which is different from above because we use more G vectors

    n_rgrid1, n_rgrid2, n_rgrid3 = [int(np.round(len(fR) ** (1 / 3)))] * 3
    fR3D = np.reshape(fR, (n_rgrid3, n_rgrid2, n_rgrid1))
    fR3D = fR3D
    normalized_fG = np.ravel(np.fft.fftshift(np.fft.fftn(fR3D)) / len(rlist))
    max_index1, max_index2, max_index3 = (
        int(n_rgrid1 / 2),
        int(n_rgrid2 / 2),
        int(n_rgrid3 / 2),
    )
    fG = []
    if idx is None:
        for g in G_dens_int:
            g_inv = g
            ix, iy, iz = (
                int(round(g_inv[0])),
                int(round(g_inv[1])),
                int(round(g_inv[2])),
            )
            ind = (
                (max_index3 + iz) * n_rgrid2 * n_rgrid1
                + (max_index2 + iy) * n_rgrid1
                + (max_index1 + ix)
            )
            if ind <= n_rgrid1 * n_rgrid2 * n_rgrid3:
                fG.append(normalized_fG[ind])  # *np.exp(-1J*np.inner(-r_i,g)) )
            else:
                fG.append(0)
                sys.stdout = open("warning_fft.txt", "a")
                print("warning", ind, g)
                sys.stdout.close()

        return np.array(fG)
    else:
        g_inv = G_dens_int[idx]
        ix, iy, iz = int(round(g_inv[0])), int(round(g_inv[1])), int(round(g_inv[2]))
        ind = (
            (max_index3 + iz) * n_rgrid2 * n_rgrid1
            + (max_index2 + iy) * n_rgrid1
            + (max_index1 + ix)
        )
        if ind <= n_rgrid1 * n_rgrid2 * n_rgrid3:
            fG.append(normalized_fG[ind])  # *np.exp(-1J*np.inner(rvec-r_i,G_dens)) )
        else:
            fG.append(0)
            sys.stdout = open("warning_fft.txt", "a")
            print("warning", ind, g)
            sys.stdout.close()

        return np.array(fG)


def myfft_dens_vv(fG, G_dens_int, rlist, r_i):  #
    n_rgrid1, n_rgrid2, n_rgrid3 = [int(np.round(len(rlist) ** (1 / 3)))] * 3
    L = rlist[n_rgrid1 - 1][0] + rlist[1][0]
    normalized_fG = np.zeros((n_rgrid1 * n_rgrid2 * n_rgrid3)) + 0j
    max_index1, max_index2, max_index3 = (
        int(n_rgrid1 / 2),
        int(n_rgrid2 / 2),
        int(n_rgrid3 / 2),
    )
    for ifG in range(len(fG)):
        g_inv = G_dens_int[ifG]
        # HACK!!!!!
        G_dens_ifG = G_dens_int[ifG] * 2 * np.pi / L
        ix, iy, iz = int(round(g_inv[0])), int(round(g_inv[1])), int(round(g_inv[2]))
        ind = (
            (max_index3 + iz) * n_rgrid2 * n_rgrid1
            + (max_index2 + iy) * n_rgrid1
            + (max_index1 + ix)
        )
        normalized_fG[ind] = fG[ifG] * np.exp(1j * np.inner(-r_i, G_dens_ifG))
    normalized_fG3D = np.reshape(normalized_fG, (n_rgrid3, n_rgrid2, n_rgrid1))
    fR = np.ravel(np.fft.ifftn(np.fft.ifftshift(normalized_fG3D))) * len(rlist)

    return fR
