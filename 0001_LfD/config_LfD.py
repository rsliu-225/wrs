import os
import platform

ROOT = os.path.abspath(os.path.dirname(__file__))
if platform.system().lower() == 'linux':
    DATA_PATH = '/media/rsliu/Data/wrs'
else:
    # DATA_PATH = 'D:\\wrs'
    DATA_PATH = 'C:/Users/rsliu/Documents/GitHub/wrs/0002_rs/'

