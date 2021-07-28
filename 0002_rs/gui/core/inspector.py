from gui.core.widgets.base import GuiFrame
from gui.core.widgets.listbox import ListBox
from gui.core.widgets.optionmenu import OptionMenu
import config
import pickle
import copy

STATE = config.STATE


class InspectorPanel(GuiFrame):
    def __init__(self, root, inspector):
        GuiFrame.__init__(self, root=root, topmost=True, close_hidden=True)
        with open(config.ROOT + "/debuger/inspector_data.pkl", "wb") as f:
            pickle.dump(inspector, f)
        self.set_title("Inspector Panel")
        self.inspector = inspector
        self.referencetable = None
        error = inspector.get_error()
        self.add_text("Select the error type:")
        self.widgets["option_menu"] = OptionMenu(self, options=error, callback=self.update_listbox,
                                                 initial_value="select")
        self.widgets["option_menu"].grid(row=self.row, column=1, sticky='ew')
        self.widgets["list_box"] = ListBox(self, on_select_callback=self.callback, options=[], width=25, height=10,
                                           selectmode='browse')
        self.widgets["list_box"].grid(row=self.irow, column=0)

    def callback(self, value):
        value = value[0]
        batch = self.batch
        id = value.split(":")[0]
        errorinfo = self.referencetable[id]
        if STATE.get('rbthi', None):
            rbtmg = STATE['rbthi'].rbtmg
            rbt = STATE['rbthi'].rbt
            rbtball = STATE['rbthi'].rbtball
            ctcallback = STATE['rbthi'].ctcallback
            rbt.movearmfk(errorinfo.rgtarmjnts, armname="rgt")
            rbt.movearmfk(errorinfo.lftarmjnts, armname="lft")
            rbt.opengripper(armname="lft", jawwidth=errorinfo.lftjawwidth)
            rbt.opengripper(armname="rgt", jawwidth=errorinfo.rgtjawwidth)
            if batch.get("rbtcm", False):
                batch["rbtcm"].detachNode()
            if batch.get("rbtball", False):
                batch["rbtball"].detachNode()
            batch["rbtcm"] = rbtmg.genmnp(rbt)
            batch["rbtcm"].reparentTo(base.render)
            batch["rbtball"] = rbtball.showfullcn(rbt)
            batch["rbtball"].reparentTo(base.render)
            print("ERROR START")
            ctcallback.setarmname("rgt")
            print(
                f"The right hand collision status:{ctcallback.iscollided(errorinfo.rgtarmjnts, obstaclecmlist=STATE['obscmlist'])}")
            ctcallback.setarmname("lft")
            print(
                f"The lft hand collision status:{ctcallback.iscollided(errorinfo.lftarmjnts, obstaclecmlist=STATE['obscmlist'])}")
            print("ERROR STOP")
        if STATE.get('objcm', None):
            if batch.get("objcm", False):
                batch["objcm"].detachNode()
            batch["objcm"] = copy.deepcopy(STATE['objcm'])
            batch["objcm"].sethomomat(errorinfo.objmat)
            print(errorinfo.objmat)
            batch["objcm"].reparentTo(STATE["base"].render)

    def update_listbox(self, error_name):
        inspector = self.inspector
        self.widgets["list_box"].updateitems(inspector.get_error_info_name_id(error_name))
        self.referencetable = inspector.get_error_info_id(error_name)


if __name__ == '__main__':
    import tkinter as tk

    root = tk.Tk()
    child = InspectorPanel(root, )
    root.mainloop()
