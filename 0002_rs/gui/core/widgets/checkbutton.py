from .widget import Widget
import tkinter as tk


class Checkbutton(Widget):
    """
    Easy-to-use check button.  Takes most options that work with
    a normal CheckButton. Attempts to call your callback
    function - if assigned - whenever there is a change to
    the check button.::

        # create the smart spinbox and grid
        scb = SmartCheckbutton(root)
        scb.grid()

        # define a callback function that retrieves
        # the currently selected option
        def callback():
            print(scb.get())

        # add the callback function to the checkbutton
        scb.add_callback(callback)

    :param parent: the tk parent frame
    :param callback: python callable
    :param options: any options that are valid for tkinter.Checkbutton
    """

    def __init__(self, parent, text="check", callback: callable = None, **options):
        self._parent = parent
        super().__init__(self._parent)

        self._var = tk.BooleanVar()
        self._cb = tk.Checkbutton(self, variable=self._var, **options)
        self._cb.grid(row=0, column=0)

        self._label = tk.Label(self, text=text)
        self._label.grid(row=0, column=1)

        if callback is not None:
            def internal_callback(*args):
                try:
                    callback()
                except TypeError:
                    callback(self.get())

            self._var.trace('w', internal_callback)
