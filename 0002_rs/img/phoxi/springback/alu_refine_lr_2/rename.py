import os

for f in os.listdir('./'):
    print(f)
    new_f = f'{int(f.split("_")[0]) - 5}_{"_".join(f.split("_")[1:])}'
    print(new_f)
    os.rename(f, new_f)
