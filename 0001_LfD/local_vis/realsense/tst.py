import pickle

intr = pickle.load(open('realsense_intr.pkl', 'rb'))
print(intr)
# {'width': 640, 'height': 480, 'fx': 610.5982666015625, 'fy': 610.678955078125,
#  'ppx': 328.25347900390625, 'ppy': 246.1207275390625}
