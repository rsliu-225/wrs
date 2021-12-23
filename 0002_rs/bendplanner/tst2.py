import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


def dist_ponto_cj(ponto, lista):
    return [euclidean(ponto, lista[j]) for j in range(len(lista))]


def ponto_mais_longe(lista_ds):
    ds_max = max(lista_ds)
    idx = lista_ds.index(ds_max)
    return pts[idx]


N = 80
K = 40
farthest_pts = [0] * K
print('N = %d, K = %d' % (N, K))

# x=[ np.random.randint(1,N) for p in range(N)]
# y=[ np.random.randint(1,N) for p in range(N)]
x = np.random.random_sample((N,))
y = np.random.random_sample((N,))
pts = [[x[i], y[i]] for i in range(N)]

P0 = pts[np.random.randint(0, N)]
farthest_pts[0] = P0
ds0 = dist_ponto_cj(P0, pts)

ds_tmp = ds0
# print ds_tmp
for i in range(1, K):
    # PML =
    farthest_pts[i] = ponto_mais_longe(ds_tmp)
    ds_tmp2 = dist_ponto_cj(farthest_pts[i], pts)
    ds_tmp = [min(ds_tmp[j], ds_tmp2[j]) for j in range(len(ds_tmp))]
    print('P[%d]: %s' % (i, farthest_pts[i]))

# print farthest_pts

xf = [farthest_pts[j][0] for j in range(len(farthest_pts))]
yf = [farthest_pts[j][1] for j in range(len(farthest_pts))]

fig, ax = plt.subplots()
plt.grid(False)
plt.scatter(x, y, c='k', s=4)
plt.scatter(xf, yf, c='r', s=4)
plt.show()
