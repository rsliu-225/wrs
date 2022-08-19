import pickle
import os
import config

if __name__ == '__main__':
    fo = 'stick'
    f = 'penta'

    textureimg, _, pcd = pickle.load(open(os.path.join(config.ROOT, 'img/phoxi', fo, f), 'rb'))
    goal_pseq = pickle.load(open(os.path.join(config.ROOT, f'bendplanner/goal/pseq/{f}.pkl'), 'rb'))
