import os
import platform

ROOT = os.path.abspath(os.path.dirname(__file__))
if platform.system().lower() == 'linux':
    DATA_PATH = '/media/rsliu/Data/wrs'
else:
    DATA_PATH = 'D:\\wrs'

