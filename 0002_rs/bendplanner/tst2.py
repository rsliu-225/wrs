from scipy.interpolate import LinearNDInterpolator
import numpy as np

# sample array for field components
Ex1 = np.array([8.84138516e+01, 8.84138516e+01, 7.77498363e+01, 5.77080432e+01])
Ey1 = np.array([1.54844696e+02, 1.54844696e+02, 1.36168141e+02, 1.01067698e+02])
Ez1 = np.array([-2.45922135e+03 - 2.45922135e+03 - 2.45922135e+03 - 2.45922135e+03])

# sample array for position
x = np.array([1.94871844, 5.61111111, 8.59672097, 10.54543941])
y = np.array([8.84138516e+01, 8.84138516e+01, 7.77498363e+01, 5.77080432e+01])
z = np.array([30.55555556, 30.55555556, 30.55555556, 30.55555556])

# linear interpolation of Ex1, Ey1, Ez1
Exf = LinearNDInterpolator((x, y, z), Ex1)
Eyf = LinearNDInterpolator((x, y, z), Ey1)
Ezf = LinearNDInterpolator((x, y, z), Ez1)

# array of new point
x1 = np.linspace(0, 5, 10)
y1 = np.linspace(0, 7, 10)
z1 = np.linspace(0, 10, 10)

# creating array([x1,y1,z1],[x2,y2,z2],....) for new grids
X = np.dstack((x1, y1, z1))
points = np.array(X)

# Field at new grids after linear interpolation
fEx = Exf(points)
fEy = Eyf(points)
fEz = Ezf(points)
print(fEx, fEy, fEz)
