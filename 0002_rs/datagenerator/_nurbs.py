from geomdl import NURBS
import matplotlib.pyplot as plt
import numpy as np
# Create a 3-dimensional B-spline Curve
curve = NURBS.Curve()

# Set degree
curve.degree = 3

# Set control points (weights vector will be 1 by default)
# Use curve.ctrlptsw is if you are using homogeneous points as Pw
curve.ctrlpts = [[10, 5, 10], [10, 20, -30], [40, 10, 25], [-10, 5, 0]]

# Set knot vector
curve.knotvector = [0, 0, 0, 0, 1, 1, 1, 1]

# Set evaluation delta (controls the number of curve points)
curve.delta = 0.05

# Get curve points (the curve will be automatically evaluated)
curve_points = np.asarray(curve.evalpts)
ctrlpts = np.asarray(curve.ctrlpts )
ax = plt.figure().add_subplot(projection='3d')
print(curve_points)
ax.plot(curve_points[:, 0], curve_points[:, 1], curve_points[:, 2])
ax.plot(ctrlpts[:, 0], ctrlpts[:, 1], ctrlpts[:, 2])
plt.show()
