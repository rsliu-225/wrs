import os
import pickle

import numpy as np

import config
from localenv import envloader as el
import manipulation.grip.robotiqhe.robotiqhe as rtqhe
import robotcon.ur3edual as ur3ex
import robotsim.robots.dualarm.ur3edual.ur3edual as ur3es
import robotsim.robots.dualarm.ur3edual.ur3edualmesh as ur3esm


def load_phoxicalibmat(amat_path=os.path.join(config.ROOT, "./camcalib/data/phoxi_calibmat.pkl")):
    amat = pickle.load(open(amat_path, "rb"))
    return amat


def load_kntcalibmat(amat_path=os.path.join(config.ROOT, "./camcalib/data/knt_calibmat.pkl")):
    amat = pickle.load(open(amat_path, "rb"))
    return amat


def transform_pcd(pcd, transmat):
    homopcd = np.ones((4, len(pcd)))
    homopcd[:3, :] = pcd.T
    realpcd = np.dot(transmat, homopcd).T
    return realpcd[:, :3]


def show_pcd_in_realenv(pcd, realrbt=False):
    hndfa = rtqhe.HandFactory()
    rgthnd = hndfa.genHand()
    lfthnd = hndfa.genHand()
    rbts = ur3es.Ur3EDualRobot(rgthnd=rgthnd, lfthnd=lfthnd)
    rbtsmg = ur3esm.Ur3EDualMesh()
    env = el.Env_wrs(boundingradius=7.0)
    env.reparentTo(base.render)

    if realrbt:
        rbtx = ur3ex.Ur3EDualUrx(hndfa)

        for armname in ["lft", "rgt"]:
            tmprealjnts = rbtx.getjnts(armname)
            rbts.movearmfk(tmprealjnts, armname)

    rbtsmg.genmnp(rbts).reparentTo(base.render)

    pcddnp = base.pg.genpointcloudnp(pcd)
    pcddnp.reparentTo(base.render)
