import numpy as np
import time
import cv2
from direct.showbase.ShowBase import ShowBase
from panda3d.core import FrameBufferProperties, WindowProperties
from panda3d.core import GraphicsPipe, GraphicsOutput
from panda3d.core import Texture
from panda3d.core import loadPrcFileData
import os
import modeling.collision_model as cm

loadPrcFileData('', 'show-frame-rate-meter true')
loadPrcFileData('', 'sync-video 0')

class Env_wrs(object):
    def __init__(self, boundingradius=.001, betransparent=False):
        """
        load obstacles model
        separated by category

        :param base:
        author: weiwei
        date: 20181205
        """

        self.__this_dir, _ = os.path.split(__file__)

        # table
        self.__tablepath = os.path.join(self.__this_dir, "../obstacles", "ur3edtable.stl")
        self.__tablecm = cm.CollisionModel(self.__tablepath, expand_radius=boundingradius, btransparency=betransparent)
        self.__tablecm.set_pos([.18, 0, 0])
        self.__tablecm.set_rgba([.32, .32, .3, 1.0])

        self.__battached = False
        self.__changableobslist = []

    def reparentTo(self, nodepath):
        if not self.__battached:
            # table
            self.__tablecm.attach_to(nodepath)
            # housing
            self.__battached = True

    def loadobj(self, name):
        self.__objpath = os.path.join(self.__this_dir, "../../0000_srl/objects", name)
        self.__objcm = cm.CollisionModel(self.__objpath, cdprimit_type="ball")
        return self.__objcm

    def getstationaryobslist(self):
        """
        generate the collision model for stationary obstacles

        :return:

        author: weiwei
        date: 20180811
        """

        stationaryobslist = [self.__tablecm]
        return stationaryobslist

    def getchangableobslist(self):
        """
        get the collision model for changable obstacles

        :return:

        author: weiwei
        date: 20190313
        """
        return self.__changableobslist

    def addchangableobs(self, nodepath, objcm, pos, rot):
        """

        :param objcm: CollisionModel
        :param pos: nparray 1x3
        :param rot: nparray 3x3
        :return:

        author: weiwei
        date: 20190313
        """

        self.__changableobslist.append(objcm)
        objcm.attach_to(nodepath)
        objcm.setMat(base.pg.npToMat4(rot, pos))

    def addchangableobscm(self, objcm):
        self.__changableobslist.append(objcm)

    def removechangableobs(self, objcm):
        if objcm in self.__changableobslist:
            objcm.remove()

def show_rgbd_image(image, depth_image, delay=1, depth_offset=0.0, depth_scale=1.0):
    if depth_image.dtype != np.uint8:
        if depth_scale is None:
            depth_scale = depth_image.max() - depth_image.min()
        if depth_offset is None:
            depth_offset = depth_image.min()
        depth_image = np.clip((depth_image - depth_offset) / depth_scale, 0.0, 1.0)
        depth_image = (255.0 * depth_image).astype(np.uint8)
    depth_image = np.tile(depth_image, (1, 1, 3))
    if image.shape[2] == 4:  # add alpha channel
        alpha = np.full(depth_image.shape[:2] + (1,), 255, dtype=np.uint8)
        depth_image = np.concatenate([depth_image, alpha], axis=-1)
    images = np.concatenate([image, depth_image], axis=1)
    # images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # not needed since image is already in BGR format
    cv2.imshow('Image window', images)
    key = cv2.waitKey(delay)
    key &= 255
    if key == 27 or key == ord('q'):
        print("Pressed ESC or q, exiting")
        exit_request = True
    else:
        exit_request = False
    return exit_request


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Load the environment model.
        self.scene = Env_wrs()
        self.scene.reparentTo(self.render)
        print(self.scene)

        base.run()
        self.scene = self.loader.loadModel("models/environment")
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Needed for camera image
        self.dr = self.camNode.getDisplayRegion(0)

        # Needed for camera depth image
        winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
        fbprops = FrameBufferProperties()
        fbprops.setDepthBits(1)
        self.depthBuffer = self.graphicsEngine.makeOutput(
            self.pipe, "depth buffer", -2,
            fbprops, winprops,
            GraphicsPipe.BFRefuseWindow,
            self.win.getGsg(), self.win)
        self.depthTex = Texture()
        self.depthTex.setFormat(Texture.FDepthComponent)
        self.depthBuffer.addRenderTexture(self.depthTex,
                                          GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
        lens = self.cam.node().getLens()
        # the near and far clipping distances can be changed if desired
        # lens.setNear(5.0)
        # lens.setFar(500.0)
        self.depthCam = self.makeCamera(self.depthBuffer,
                                        lens=lens,
                                        scene=render)
        self.depthCam.reparentTo(self.cam)

        # TODO: Scene is rendered twice: once for rgb and once for depth image.
        # How can both images be obtained in one rendering pass?

    def get_camera_image(self, requested_format=None):
        """
        Returns the camera's image, which is of type uint8 and has values
        between 0 and 255.
        The 'requested_format' argument should specify in which order the
        components of the image must be. For example, valid format strings are
        "RGBA" and "BGRA". By default, Panda's internal format "BGRA" is used,
        in which case no data is copied over.
        """
        tex = self.dr.getScreenshot()
        if requested_format is None:
            data = tex.getRamImage()
        else:
            data = tex.getRamImageAs(requested_format)
        image = np.frombuffer(data, np.uint8)  # use data.get_data() instead of data in python 2
        image.shape = (tex.getYSize(), tex.getXSize(), tex.getNumComponents())
        image = np.flipud(image)
        return image

    def get_camera_depth_image(self):
        """
        Returns the camera's depth image, which is of type float32 and has
        values between 0.0 and 1.0.
        """
        data = self.depthTex.getRamImage()
        depth_image = np.frombuffer(data, np.float32)
        depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
        depth_image = np.flipud(depth_image)
        return depth_image


def convert_depth2pcd(depthnarray):
    h, w, _ = depthnarray.shape
    y_ = np.linspace(1, h, h)
    x_ = np.linspace(1, w, w)
    mesh_x, mesh_y = np.meshgrid(x_, y_)
    z_ = depthnarray.flatten() * 1000
    pcd = np.zeros((np.size(mesh_x), 3))
    pcd[:, 0] = np.reshape(mesh_x, -1)
    pcd[:, 1] = np.reshape(mesh_y, -1)
    pcd[:, 2] = np.reshape(z_, -1)
    return np.delete(pcd, np.where(pcd[:, 2] == 0)[0], axis=0)


if __name__ == '__main__':
    import pickle

    app = MyApp()

    frames = 1800
    radius = 20
    step = 0.1
    start_time = time.time()

    for t in range(frames):
        angleDegrees = t * step
        angleRadians = angleDegrees * (np.pi / 180.0)
        app.cam.setPos(radius * np.sin(angleRadians), -radius * np.cos(angleRadians), 3)
        app.cam.setHpr(angleDegrees, 0, 0)
        app.graphicsEngine.renderFrame()
        image = app.get_camera_image()
        depth_image = app.get_camera_depth_image()
        pcd = convert_depth2pcd(depth_image)
        pcd = pcd / 100

        # pickle.dump(pcd, open('tst.pkl', 'wb'))
        # break
        show_rgbd_image(image, depth_image)

    end_time = time.time()
    print("average FPS: {}".format(frames / (end_time - start_time)))
