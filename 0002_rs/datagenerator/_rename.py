import os

path = 'E:/liu/org_data/dataset'
cat = 'multiview_true'

for fo in os.listdir(os.path.join(path, cat)):
    print(fo)
    for f in os.listdir(os.path.join(path, cat, fo)):
        if f[:4] == 'rand':
            print(f, 'rlen' + f[4:])
            os.rename(os.path.join(path, cat, fo, f), os.path.join(path, cat, fo, 'rlen' + f[4:]))
            print(os.path.exists(os.path.join(path, cat, fo, 'rlen' + f[4:])))
