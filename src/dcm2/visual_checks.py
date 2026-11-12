
import matplotlib.pyplot as plt
import numpy as np

import splines

def main():
    x = np.linspace(0, 1, 5000)
    knots = [0., .2, .4, .6, .8, 1.]

    mspline_basis = splines.evaluate('M', x, deg:=3, knots)
    ispline_basis = splines.evaluate('I', x, deg:=3, knots)

    fig, axes = plt.subplots(
        nrows=2,
        layout='constrained',
        )
    for ax in axes[:]:
        ax.vlines([0] + knots + [1], 0, 20, color='k', ls=':')

    axes[0].set_title(f"M-spline basis of degree {deg}")
    axes[0].plot(x, mspline_basis.T)
    axes[0].plot(x, np.sum(mspline_basis, axis=0), 'k--')

    axes[1].set_title(f"I-spline basis of degree {deg}")
    axes[1].plot(x, ispline_basis.T)
    axes[1].plot(x, np.sum(ispline_basis, axis=0), 'k--')

    plt.show()

if __name__ == '__main__':
    main()
