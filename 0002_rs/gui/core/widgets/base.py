import tkinter as tk
from tkinter.font import Font
import config

STATE = config.STATE


class GuiFrame(tk.Toplevel):
    super_batch = {}

    def __init__(self, root, topmost=False, close_hidden=False, resizable=False):

        tk.Toplevel.__init__(self, root)
        self._close_hidden = close_hidden
        self._topmost = topmost
        self._resizable = resizable
        self.widgets = {}
        self.batch = {}
        self.__row = 0

    @property
    def irow(self):
        row = self.__row
        self.__row += 1
        return row

    @property
    def row(self):
        row = self.__row - 1
        return max(row, 0)

    def add_title(self, text, row=None, column=0):
        fontStyle = Font(family="Lucida Grande", size=13)
        label_widget = tk.Label(self, justify=tk.LEFT, text=text, font=fontStyle)
        label_widget.grid(row=row if isinstance(row, int) else self.irow, column=column)
        return label_widget

    def add_text(self, text, row=None, column=0):
        label_widget = tk.Label(self, text=text)
        label_widget.grid(row=row if isinstance(row, int) else self.irow, column=column)
        return label_widget

    def set_title(self, title):
        self.title(title)

    def set_size(self, width, height):
        self.geometry(f"{int(width)}x{int(height)}")

    def registeration(self):
        # self.tk.pack_slaves()
        if self._close_hidden:
            self.protocol("WM_DELETE_WINDOW", self.on_closing)
        if self._topmost:
            self.attributes("-topmost", 1)
        if self._resizable:
            self.resizable(True, True)
        else:
            self.resizable(False, False)

    def on_closing(self):
        self.withdraw()

    def add(self):
        pass

    def addtask(self, task, args=None, timestep=0.1):
        runparameter = {
            "funcOrTask": task, "name": task.__code__.co_name,
        }
        run = STATE['taskMgr'].add
        if timestep > 0:
            runparameter.setdefault("delayTime", timestep)
            run = STATE['taskMgr'].doMethodLater

        if args is not None:
            runparameter.setdefault("extraArgs", args)
            runparameter.setdefault("appendTask", True)

        run(**runparameter)


if __name__ == '__main__':
    root = tk.Tk()
    child = GuiFrame(root)
    child.set_size(width=400, height=200)
    child.set_title("Hello world")
    child.add_title("sdfdsf")
    root.mainloop()
