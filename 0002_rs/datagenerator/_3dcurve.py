from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def spl_inp(kpts,k=3):
    kpts = kpts.transpose()
    # now we get all the knots and info about the interpolated spline
    tck, u = interpolate.splprep(kpts, k=k)
    # here we generate the new interpolated dataset,
    # increase the resolution by increasing the spacing, 500 in this example
    new = interpolate.splev(np.linspace(0, 1, 500), tck, der=0)
    pts = np.asarray(new).transpose()
    print(pts)

    # now lets plot it!
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(kpts[0], kpts[1], kpts[2], label='originalpoints', lw=2, c='Dodgerblue')
    ax.plot(new[0], new[1], new[2], label='fit', lw=2, c='red')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    kpts = np.asarray([[0, 0, 0], [.04, -.02, -0.01], [.08, .5, -0.02], [.1, .1, -0.02], [.2, .2, 0]])
    spl_inp(kpts)
