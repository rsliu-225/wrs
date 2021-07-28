PRJ_INFO = {
    'bucket': {'draw_center': (0, 100, 60), 'prj_direction': (0, -1, 0)},
    'box': {'draw_center': (140, 250, 50), 'prj_direction': (0, -1, 0)},  # circle
    # 'box': {'draw_center': (90, 250, 50), 'prj_direction': (0, -1, 0)},  # star
    # 'bunny': {'draw_center': (-5, 55, 0), 'prj_direction': (0, 0, 1)},
    'bunny': {'draw_center': (0, -25, 0), 'prj_direction': (0, 0, 1)},  # circle
    # 'bunny': {'draw_center': (0, -40, 0), 'prj_direction': (0, 0, 1)},  # star
    'cube': {'draw_center': (0, 50, 0), 'prj_direction': (0, 0, 1)},  # circle
    'leg': {'draw_center': (0, 90, 0), 'prj_direction': (0, 0, 1)},  # circle

}

OBJ_POS = {
    'box': (800, 200, 780),
    'bucket': (800, 200, 780),
    'cube': (800, 400, 780)
}

# if paintingobj_f_name == 'bucket':
#     paintingobj_item.set_drawcenter((0, 100, 60))  # bucket
#     prj_direction = np.asarray((0, -1, 0))
# elif paintingobj_f_name == 'box':
#     paintingobj_item.set_drawcenter((140, 250, 50))  # box
#     prj_direction = np.asarray((0, -1, 0))
# elif paintingobj_f_name == 'bunny':
#     paintingobj_item.set_drawcenter((-5, 55, 0))  # bunny
# else:
#     prj_direction = np.asarray((0, 0, 1))
# "DRAW": +40, +15, +2, -13
