import pickle

exp_name = 'cube'
exp_id = 'rgt_org'

info = pickle.load(open(f'./{exp_name}_{exp_id}.pkl', 'rb'))

print(len(info[0]))
print(info[0][99])
print(info[0][100])
print(info[0][101])
print(info[1][100])
print(info[2][100])
print(info[3][100])
print(info[4][100])
info[0] = info[0][:101] + info[0][102:]
info[1] = info[1][:101] + info[1][102:]
info[2] = info[2][:101] + info[2][102:]
info[3] = info[3][:101] + info[3][102:]
info[4] = info[4][:101] + info[4][102:]
print(len(info[0]))
# print(info[0][123])
# print(info[0][124])


pickle.dump(info, open(f'./{exp_name}_{exp_id}.pkl', 'wb'))

