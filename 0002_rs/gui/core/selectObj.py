from gui.base import *
from panda3d.core import CollisionNode, GeomNode, CollisionRay, CollisionTraverser, CollisionHandlerQueue, NodePath


def moverbt(bindkey, tasktable, task):
    for key in bindkey:
        if STATE['base'].inputmgr.keymap[key]:
            tasktable[bindkey.index(key)]()
            STATE['base'].inputmgr.keymap[key] = False
    return task.again


class SelectObj:
    def __init__(self):
        self.objselected = None
        self.colorsaved = None
        self.bindkey = []
        self.tasktable = []
        self.picker_cn = CollisionNode('selection')
        self.picker_cn.setFromCollideMask(GeomNode.getDefaultCollideMask())
        self.pickerRay = CollisionRay()
        self.picker_cn.addSolid(self.pickerRay)
        picker_np = STATE['base'].cam.attachNewNode(self.picker_cn)

        self.traverser = CollisionTraverser('traverser name')
        self.collisionhandler = CollisionHandlerQueue()
        self.traverser.addCollider(picker_np, self.collisionhandler)

    def selectobj(self):
        if self.objselected is not None: return
        mpos = STATE['base'].mouseWatcherNode.getMouse()
        self.pickerRay.setFromLens(STATE['base'].cam.node(), mpos.getX(), mpos.getY())
        self.traverser.traverse(STATE['base'].render)
        # Assume for simplicity's sake that myHandler is a CollisionHandlerQueue.
        if self.collisionhandler.getNumEntries() > 0:
            # This is so we get the closest object
            self.collisionhandler.sortEntries()
            pickedObj = self.collisionhandler.getEntry(0).getIntoNodePath().getParent()
            if pickedObj in self.avaliableobj:
                self.objselected = STATE['objcmlist'][self.avaliableobj.index(pickedObj)]
                self.colorsaved = self.objselected.getColor()
                self.objselected.setColor(1, 0, 0, 1)
                self.objselected.showlocalframe()
                self.passInfo()
            # print(pickedObj)

    def cancelselection(self):
        if self.objselected is None: return
        self.removeInfo()
        color = self.colorsaved
        self.objselected.setColor(color[0], color[1], color[2], color[3])
        self.objselected.unshowlocalframe()
        self.objselected = None

    def registeration(self):
        self.avaliableobj = []
        for cm in STATE['objcmlist']:
            self.avaliableobj.append(cm.objnp)
        self.bindkey = ["mouse3", "escape"]
        addkey(self.bindkey)
        self.tasktable = [
            self.selectobj,
            self.cancelselection
        ]

    def passInfo(self):
        if self.objselected is None: return
        STATE['tk'].loadValue(**{
            "position": self.objselected.getPos(),
            "rpy": self.objselected.getRPY(),
            "color": self.objselected.getColor(),
            "objselected": self.objselected,
            "host": self
        })

    def removeInfo(self):
        if self.objselected is None: return
        STATE['tk'].removestatus()

    def add(self):
        addtask(moverbt, args=[self.bindkey, self.tasktable], timestep=0.1)
