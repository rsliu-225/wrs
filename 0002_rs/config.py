import os

'''
param
'''
ID_DICT = {
    "0524": [13, 52],
    "0525": [2, 401],
    "cube": [33, 0],
    "cylinder_cad": [1, 8],
    "force": [1, 8],
    "cylinder_pcd": [3, 10],
    "helmet": [1, 8],
    "raft": [10, 176],
    "leg": [10, 104],
    "bunny": [32, 10],
    "temp": [34, 293],
    "cylinder_mtp": [55, 290],
    "bucket": [25, 295],
    "box": [25],
}

PHOXI_HOST = "10.0.1.191:18300"
ROOT = os.path.abspath(os.path.dirname(__file__))
# AMAT_F_NAME = "phoxi_calibmat_0217.pkl"
# AMAT_F_NAME = "phoxi_calibmat_0615.pkl"
# AMAT_F_NAME = "phoxi_calibmat_1222.pkl"
AMAT_F_NAME = "phoxi_calibmat_210527.pkl"
# PEN_STL_F_NAME = "pentip_short.stl"
PEN_STL_F_NAME = "pentip.stl"
IPURX = '10.0.2.11'

PREGRASP_REL_PATH = ROOT + "/graspplanner/pregrasp/"
GRASPMAP_REL_PATH = ROOT + "/graspplanner/graspmap/"
# GRASPMAP_REL_PATH = ROOT + "/graspplanner/graspmap/temp/"
MOTIONSCRIPT_REL_PATH = ROOT + "/motionscript/"
PENPOSE_REL_PATH = ROOT + "/log/penpose/"
