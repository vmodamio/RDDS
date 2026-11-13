
import matplotlib.pyplot as plt
import numpy as np

import splines

def main():
    x = np.linspace(0, 1, 5001)
    knots = [0., .2, .4, .6, .8, 1.]

    mspline = splines.evaluate('M', x, deg:=3, knots)
    ispline = splines.evaluate('I', x, deg:=3, knots)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        layout='constrained',
        )
    for ax in axes.ravel():
        for t in [0] + knots + [1]:
            ax.axvline(t, color='k', ls=':')

    axes[0, 0].set_title(f"M-spline basis of degree {deg}")
    axes[0, 0].plot(mspline.x, mspline.basis.T)
    axes[0, 0].plot(mspline.x, np.sum(mspline.basis, axis=0), 'k--')

    axes[1, 0].set_title(f"M-spline derivative of degree {deg}")
    axes[1, 0].plot(mspline.x, mspline.derivative.T)
    axes[1, 0].plot(mspline.x, np.sum(mspline.derivative, axis=0), 'k--')

    axes[0, 1].set_title(f"I-spline basis of degree {deg}")
    axes[0, 1].plot(x, ispline.basis.T)
    axes[0, 1].plot(x, np.sum(ispline.basis, axis=0), 'k--')

    axes[1, 1].set_title(f"I-spline derivative of degree {deg}")
    axes[1, 1].plot(x, ispline.derivative.T)
    axes[1, 1].plot(x, np.sum(ispline.derivative, axis=0), 'k--')

    plt.show()

if __name__ == '__main__':
    main()
