from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
from gui.base import *
import numpy as np
import utiltools.robotmath as rm


def graspobj():
    if STATE['grasp'] is not None:
        STATE['grasp'] = None
    else:
        tmp = []
        tmp2 = []
        for objcm in STATE['objcmlist']:
            pos = objcm.getPos()
            tcppos, tcprot = STATE['rbthi'].rbt.getee(armname=STATE['armname'])
            tmp.append(np.linalg.norm(tcppos - pos))
            tmp2.append([pos, objcm.gethomomat()[:3, :3]])
        STATE['grasp'] = STATE['objcmlist'][np.argmin(tmp)]
        STATE['rel'] = STATE['rbthi'].rbt.getinhandpose(*tmp2[np.argmin(tmp)], armname=STATE["armname"])
        print(" The object:", STATE['grasp'])


def moverbt(bindkey, tasktable, task):
    for key in bindkey:
        if STATE['base'].inputmgr.keymap[key]:
            tasktable[bindkey.index(key)]()
            if key == 'shift' or key == "control" or key == "space":
                STATE['base'].inputmgr.keymap[key] = False
    return task.again


class Move:
    def __init__(self):
        self.onscreentext = {}
        self.bindkey = []
        self.tasktable = []

    def showinfo(self):
        self.onscreentext['text'] = OnscreenText(
            text=str("Mode: %d." % (STATE['mode'])), pos=(-.9, -.9 + .05, 0),
            scale=0.05,
            fg=(0., 0, 0, 1),
            align=TextNode.ALeft, mayChange=1)
        self.onscreentext['text2'] = OnscreenText(
            text=str("The status of the robot is: Free"), pos=(-.9, -.9, 0),
            scale=0.05,
            fg=(0, 0, 0, 1),
            align=TextNode.ALeft, mayChange=1)
        self.onscreentext['text3'] = OnscreenText(
            text=str("The armname is: Right"), pos=(-.9, -.9 + .1, 0),
            scale=0.05,
            fg=(0, 0, 0, 1),
            align=TextNode.ALeft, mayChange=1)

    def switcharm(self):
        STATE['armname'] = 'rgt' if STATE['armname'] == "lft" else "lft"
        self.onscreentext['text3']['text'] = self.onscreentext['text3']['text'][:15] + ["Right", "Left"][
            0 if STATE['armname'] == 'rgt' else 1]

    def switchmode(self):
        STATE['mode'] = STATE['mode'] + 1 if STATE['mode'] < 2 else 0
        self.onscreentext['text']['text'] = replaceat(self.onscreentext['text']['text'], 6, STATE['mode'])

    def movefuncfactory(self, movedirection, rotationaxis=None, distance=5, name="function"):
        movedirection = np.array(movedirection)

        def func():
            STATE['rbthi'].ctcallback.setarmname(armname=STATE['armname'])
            tcppos, tcprot = STATE['rbthi'].rbt.getee(armname=STATE['armname'])
            if STATE['mode'] == 0:
                tcpnew = tcppos + distance * movedirection
                tcprotnew = tcprot
            elif STATE['mode'] == 1:
                if rotationaxis is None:
                    return
                else:
                    tcpnew = tcppos
                    tcprotnew = np.dot(rm.rodrigues(rotationaxis, 5), tcprot)

            elif STATE['mode'] == 2:
                if rotationaxis is None:
                    return
                else:
                    tcpnew = tcppos
                    tcprotnew = np.dot(rm.rodrigues(np.dot(tcprot, rotationaxis), 5), tcprot)

            jnts = STATE['rbthi'].rbt.numikmsc(eepos=tcpnew, eerot=tcprotnew,
                                               msc=STATE['rbthi'].rbt.getarmjnts(armname=STATE['armname']),
                                               armname=STATE['armname'])

            if jnts is None:
                print("Cannot move")
                return

            if STATE['grasp'] is not None:
                state = int(
                    STATE['rbthi'].ctcallback.iscollidedHold(jnts, objcmlist=[STATE['grasp']],
                                                             relmatlist=[STATE['rel']],
                                                             obstaclecmlist=STATE['obscmlist']))

            else:
                state = int(STATE['rbthi'].ctcallback.iscollided(jnts, obstaclecmlist=STATE['obscmlist']))

            self.onscreentext['text2']['text'] = self.onscreentext['text2']['text'][:28] + ["Free", "Collided"][state]
            STATE['rbthi'].rbt.movearmfk(jnts, armname=STATE['armname'])
            if STATE['grasp'] is not None:
                STATE['grasp'].sethomomat(
                    rm.homobuild(*STATE['rbthi'].rbt.getworldpose(*STATE['rel'], armname=STATE["armname"])))
            STATE['rbtmg_instance'].detachNode()
            STATE['rbtmg_instance'] = STATE['rbthi'].rbtmg.genmnp(STATE['rbthi'].rbt)
            STATE['rbtmg_instance'].reparentTo(STATE['base'].render)
            STATE['rbtball_instance'] = STATE['rbthi'].rbtball.genfullbcndict(STATE['rbthi'].rbt)
            STATE['rbthi'].rbtball.showcn(STATE['rbtball_instance'])

        func.__name__ = name
        return func

    def registeration(self):
        # adding keymap
        STATE['mode'] = 0
        STATE['grasp'] = None
        self.bindkey = ['w', 's', 'a', 'd', 'q', 'e', 'control', 'shift', 'space']
        self.showinfo()
        addkey(self.bindkey)
        self.tasktable = [
            self.movefuncfactory([1, 0, 0], rotationaxis=[0, 1, 0], name=self.bindkey[0]),
            self.movefuncfactory([-1, 0, 0], rotationaxis=[0, -1, 0], name=self.bindkey[1]),
            self.movefuncfactory([0, 1, 0], rotationaxis=[0, 0, 1], name=self.bindkey[2]),
            self.movefuncfactory([0, -1, 0], rotationaxis=[0, 0, -1], name=self.bindkey[3]),
            self.movefuncfactory([0, 0, 1], rotationaxis=[1, 0, 0], name=self.bindkey[4]),
            self.movefuncfactory([0, 0, -1], rotationaxis=[-1, 0, 0], name=self.bindkey[5]),
            self.switcharm,
            self.switchmode,
            graspobj
        ]

    def add(self):
        addtask(moverbt, [self.bindkey, self.tasktable], 0.05)
