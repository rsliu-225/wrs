import numpy as np


def projection(normal, vector):
    return vector - normal * np.dot(vector, normal) / np.linalg.norm(normal)


def saveimg(imgname, xsize=1600, ysize=1200):
    # base.graphicsEngine.readyFlip()

    # winprops = WindowProperties.size(xsize, ysize)
    # props = FrameBufferProperties()
    # props.setRgbColor(1)
    # props.setAlphaBits(1)
    # props.setDepthBits(1)
    # myBuff = base.graphicsEngine.makeOutput(
    #     base.pipe, "offscreenBuffer",
    #     -2, props, winprops,
    #     GraphicsPipe.BFRefuseWindow,
    #     base.win.getGsg(), base.win)

    # myBuff.saveScreenshot(base.pg.Filename(imgname+".jpg"))
    base.graphicsEngine.renderFrame()
    base.win.saveScreenshot(base.pg.Filename(imgname + ".bmp"))
    # myBuff.setActive(False)
    # image = PNMImage()
    # dr = base.camNode.getDisplayRegion(0)
    # dr.getScreenshot(image)
    # image.write(Filename(imgname+".bmp"))


def zoombase(direction=np.array([1, 1, 1])):
    bounds = base.render.getTightBounds()
    if bounds is not None:
        center = (bounds[0] + bounds[1]) / 2
        # print(center)
        point1 = np.array([bounds[0][0], bounds[0][1], bounds[0][2]])
        point2 = np.array([bounds[1][0], bounds[1][1], bounds[1][2]])
        point1_project = projection(direction, point1)
        point2_project = projection(direction, point2)
        line_project = point2_project - point1_project
        length = np.linalg.norm(line_project) / 2
        # print(length)
        # horizontal_vector = np.dot(R_matrix,horizontal_vector_origin)
        # vertical__vector = np.dot(R_matrix, vertical_vector_origin)

        # project to the plane
        Fov = base.cam.node().getLens().getFov()
        horizontalAngle, verticalAngle = Fov[0], Fov[1]
        d_horzontal = length / np.tan(np.deg2rad(horizontalAngle) / 2)
        d_vertical = length / np.tan(np.deg2rad(verticalAngle) / 2)
        # angle = [horizontalAngle, verticalAngle][np.argmax([d_horzontal,d_vertical])]
        distance = max(d_horzontal, d_vertical)
        # print(distance)

        campos = np.array([center[0], center[1], center[2]]) + direction * distance
        base.cam.setPos(campos[0], campos[1], campos[2])
        base.cam.lookAt(center[0], center[1], center[2])


def clearbase(sceneflag=3):
    print(len(base.render.children))
    for i in base.render.children:
        if sceneflag > 0:
            sceneflag -= 1
        else:
            if str(i).split('/')[-1] != 'defaultname':
                i.removeNode()


def clearobj_by_name(namelist):
    for i in base.render.children:
        if str(i).split('/')[-1] in namelist:
            i.removeNode()
