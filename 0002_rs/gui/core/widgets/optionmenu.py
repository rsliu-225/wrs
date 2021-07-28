import tkinter as tk
from .widget import Widget


class OptionMenu(Widget):
    """
    Classic drop down entry with built-in tracing variable.::

        # create the dropdown and grid
        som = SmartOptionMenu(root, ['one', 'two', 'three'])
        som.grid()

        # define a callback function that retrieves
        # the currently selected option
        def callback():
        print(som.get())

        # add the callback function to the dropdown
        som.add_callback(callback)

    :param data: the tk parent frame
    :param options: a list containing the drop down options
    :param initial_value: the initial value of the dropdown
    :param callback: a function
    """

    def __init__(self, parent, options: list, initial_value: str = None,
                 callback: callable = None, callbackparameter: dict = None):
        super().__init__(parent)

        self._var = tk.StringVar()
        if len(options) < 1:
            options = ["None"]

        self.__initial_value = initial_value
        self._var.set(initial_value if initial_value else options[0])

        self.option_menu = tk.OptionMenu(self, self._var,
                                         *options)
        self.option_menu.grid(row=0, column=0)

        if callback is not None:
            def internal_callback(*args):
                try:
                    callback()
                except TypeError:
                    callback(self.get())

            self._var.trace('w', internal_callback)

    def updateoptions(self, items):
        # Reset var and delete all old options
        self._var.set(self.__initial_value if self.__initial_value else items[0])
        self.option_menu['menu'].delete(0, 'end')

        # Insert list of new options (tk._setit hooks them up to var)
        for choice in items:
            self.option_menu['menu'].add_command(label=choice, command=tk._setit(self._var, choice))
