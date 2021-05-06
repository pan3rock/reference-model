import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d


def main(z, dz, vs):
    z1, z2 = z
    dz1, dz2 = dz
    n1 = round(z1 / dz1)
    z1_new = np.arange(n1) * dz1
    n2 = round(2.0 * (z2 - z1_new[-1]) / (dz1 + dz2)) + 1
    dz_new = np.linspace(dz1, dz2, n2)
    z2_new = []
    z_cum = z1_new[-1]
    for i in range(n2):
        z_cum += dz_new[i]
        z2_new.append(z_cum)
    z2_new = np.asarray(z2_new)
    z_new = np.concatenate([z1_new, z2_new])

    vs1, vs2 = vs
    intp = interp1d([0, z2], vs, fill_value='extrapolate')
    vs_new = intp(z_new)

    def vs2vp(vs):
        return 1.7321 * vs

    def vp2rho(vp):
        return 1.741 * vp**0.25

    vp_new = vs2vp(vs_new)
    rho_new = vp2rho(vp_new)

    nl = vs_new.shape[0]
    model = np.zeros([nl, 5])
    model[:, 0] = np.arange(nl) + 1
    model[:, 1] = z_new
    model[:, 2] = rho_new
    model[:, 3] = vs_new
    model[:, 4] = vp_new

    plt.figure()
    plt.step(vs_new, z_new)
    plt.gca().invert_yaxis()
    plt.show()

    np.savetxt('model_ref.txt', model, fmt='%10.0f%10.5f%10.5f%10.5f%10.5f')


if __name__ == '__main__':
    msg = "generate reference model for near-surface case"
    parser = argparse.ArgumentParser(description=msg)
    msg = 'depth of lower bound of uniform part and gradient part'
    parser.add_argument('--z', type=float, nargs=2, help=msg)
    msg = 'layer thickness of upper and lower bound of gradient part'
    parser.add_argument('--dz', type=float, nargs=2, help=msg)
    msg = 'Vs of the surface and lower bound of gradient part'
    parser.add_argument('--vs', type=float, nargs=2, help=msg)
    args = parser.parse_args()
    z = args.z
    dz = args.dz
    vs = args.vs
    main(z, dz, vs)