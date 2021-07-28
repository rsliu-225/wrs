from gui.base import *
from tkinter import Entry, Label, LEFT, StringVar
from tkinter.font import Font
import numpy as np


class AttributePanel:
    def __init__(self):
        STATE['base'].startTk()
        self.objselected = None
        self.widgets = {}
        self.tasktable = []
        # grid layout
        self.host = None
        self.row = 0

        self.tk = STATE['base'].tkRoot
        self.tk.title("The object's property")
        self.tk.geometry("170x300")
        self.tk.resizable(False, False)
        self.tk.attributes("-topmost", 1)
        self.tk.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.addTitle("Position")
        self.addProperty("x:", name="position-x")
        self.addProperty("y:", name="position-y")
        self.addProperty("z:", name="position-z")
        self.addTitle("RPY")
        self.addProperty("r:", name="rpy-r")
        self.addProperty("p:", name="rpy-p")
        self.addProperty("y:", name="rpy-y")

    def addTitle(self, text):
        fontStyle = Font(family="Lucida Grande", size=13)
        label_widget = Label(justify=LEFT, text=text, font=fontStyle)

        self.row += 1
        label_widget.grid(row=self.row, column=0)

    def addProperty(self, label, name="", callback=None):
        label_widget = Label(text=label)
        field_text = StringVar()
        field_widget = Entry(width=8, textvariable=field_text)

        self.row += 1
        label_widget.grid(row=self.row, column=0)
        field_widget.grid(row=self.row, column=1)
        if name == "" or name in self.widgets:
            raise Exception("Error name")
        self.widgets[name] = {"label": label_widget, "field": field_widget, "field_text": field_text}

    def focusoutandentermethod(self, component, callback):
        component.bind("<FocusOut>", callback)
        component.bind("<Return>", callback)

    def setpos(self, event):
        if self.objselected is None: return
        pos = [
            float(self.widgets["position-x"]['field_text'].get()),
            float(self.widgets["position-y"]['field_text'].get()),
            float(self.widgets["position-z"]['field_text'].get())
        ]
        self.tasktable.append(
            {
                "task": self.objselected.setPos,
                "parameter": pos
            }
        )

    def setrpy(self, event):
        if self.objselected is None: return
        rpy = [
            float(self.widgets["rpy-r"]['field_text'].get()),
            float(self.widgets["rpy-p"]['field_text'].get()),
            float(self.widgets["rpy-y"]['field_text'].get())
        ]
        self.tasktable.append(
            {
                "task": self.objselected.setRPY,
                "parameter": rpy
            }
        )

    def loadValue(self, **kwargs):
        if "objselected" not in kwargs:
            return
        self.tk.deiconify()
        self.objselected = kwargs['objselected']
        if "position" in kwargs:
            x, y, z = kwargs["position"]
            self.widgets["position-x"]['field_text'].set(str(x))
            self.widgets["position-y"]['field_text'].set(str(y))
            self.widgets["position-z"]['field_text'].set(str(z))
        if "rpy" in kwargs:
            r, p, y = kwargs["rpy"]
            self.widgets["rpy-r"]['field_text'].set(str(r))
            self.widgets["rpy-p"]['field_text'].set(str(p))
            self.widgets["rpy-y"]['field_text'].set(str(y))
        if "host" in kwargs:
            self.host = kwargs["host"]

    def executetasktable(self, task):
        if len(self.tasktable) < 1: return task.again
        for t in self.tasktable:
            t["task"](*t["parameter"])
        self.tasktable = []
        return task.again

    def removestatus(self):
        self.objselected = None
        self.widgets["position-x"]['field_text'].set("")
        self.widgets["position-y"]['field_text'].set("")
        self.widgets["position-z"]['field_text'].set("")
        self.widgets["rpy-r"]['field_text'].set("")
        self.widgets["rpy-p"]['field_text'].set("")
        self.widgets["rpy-y"]['field_text'].set("")
        self.tk.withdraw()

    def on_closing(self):
        self.tk.withdraw()
        if self.host is not None:
            self.host.cancelselection()

    def registeration(self):
        self.focusoutandentermethod(self.widgets["position-x"]['field'], self.setpos)
        self.focusoutandentermethod(self.widgets["position-y"]['field'], self.setpos)
        self.focusoutandentermethod(self.widgets["position-z"]['field'], self.setpos)
        self.focusoutandentermethod(self.widgets["rpy-r"]['field'], self.setrpy)
        self.focusoutandentermethod(self.widgets["rpy-p"]['field'], self.setrpy)
        self.focusoutandentermethod(self.widgets["rpy-y"]['field'], self.setrpy)
        self.removestatus()

    def add(self):
        STATE["tk"] = self
        addtask(self.executetasktable, timestep=0.5)
