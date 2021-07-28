import glob
import os

path = './discrete/*'
i = 1

f_list = glob.glob(path)
print('変更前')
print(f_list)

for f in f_list:
    if f.find('m1') != -1:
        f_new = f.replace("m1", "m1r10")
        os.rename(f, f_new)
        i += 1

f_list = glob.glob(path)
print('変更後')
print(f_list)
