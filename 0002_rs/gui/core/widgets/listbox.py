from .widget import Widget
import tkinter as tk


class ListBox(Widget):
    """
    Easy-to-use List Box.  Takes most options that work with
    a normal CheckButton. Attempts to call your callback
    function - if assigned - whenever there is a change to
    the list box selections.::

        # create the smart spinbox and grid
        scb = SmartListBox(root, options=['one', 'two', 'three'])
        scb.grid()

        # define a callback function that retrieves
        # the currently selected option
        def callback():
            print(scb.get_selected())

        # add the callback function to the checkbutton
        scb.add_callback(callback)

    :param parent: the tk parent frame
    :param options: any options that are valid for tkinter.Checkbutton
    :param on_select_callback: python callable
    :param selectmode: the selector mode (supports "browse" and "multiple")
    """

    def __init__(self, parent, options: list,
                 width: int = 12, height: int = 5,
                 on_select_callback: callable = None,
                 selectmode: str = 'browse'):
        super().__init__(parent=parent)

        self._on_select_callback = on_select_callback
        self._values = {}  # select value

        r = 0
        self._lb = tk.Listbox(self, width=width, height=height,
                              selectmode=selectmode, exportselection=0)
        self._lb.grid(row=r, column=0, sticky='ew')
        # insert the options
        [self._lb.insert('end', option) for option in options]
        # bind the method
        self._lb.bind('<<ListboxSelect>>', lambda _: self._on_select())

        r += 1
        clear_label = tk.Label(self, text='clear', fg='blue')
        clear_label.grid(row=r, column=0, sticky='ew')
        clear_label.bind('<Button-1>', lambda _: self._clear_selected())

    def _on_select(self):
        self.after(200, self.__on_select)  # CALL the method 200ms

    def _clear_selected(self):
        for i in self._lb.curselection():
            self._lb.selection_clear(i, 'end')

        while len(self._values):
            self._values.popitem()

        if self._on_select_callback is not None:
            values = list(self._values.keys())
            try:
                self._on_select_callback(values)
            except TypeError:
                self._on_select_callback()

    def __on_select(self):
        value = self._lb.get('active')

        if self._lb.cget('selectmode') == 'multiple':
            if value in self._values.keys():
                self._values.pop(value)  # pop values
            else:
                self._values[value] = True  # value_value = True?
        else:
            while len(self._values):
                self._values.popitem()  # ?
            self._values[value] = True

        if self._on_select_callback is not None:
            values = list(self._values.keys())
            try:
                self._on_select_callback(values)
            except TypeError:
                self._on_select_callback()

    def add_callback(self, callback: callable):
        """
        Associates a callback function when the user makes a selection.

        :param callback: a callable function
        """
        self._on_select_callback = callback

    def get_selected(self):
        return list(self._values.keys())

    def select(self, value):
        options = self._lb.get(0, 'end')
        if value not in options:
            raise ValueError('Not a valid selection')

        option = options.index(value)

        self._lb.activate(option)
        self._values[value] = True

    def updateitems(self, options):
        self._lb.delete(0, "end")
        [self._lb.insert('end', option) for option in options]
