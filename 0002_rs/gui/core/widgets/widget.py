import tkinter.ttk as ttk


class Widget(ttk.Frame):
    """
    Superclass which contains basic elements of the widgets.
    """

    def __init__(self, parent):
        self._parent = parent
        super().__init__(self._parent)

        self._var = None

    def add_callback(self, callback: callable):
        """
        Add a callback on change

        :param callback: callable function
        :return: None
        """

        def internal_callback(*args):
            try:
                callback()
            except TypeError:
                callback(self.get())

        self._var.trace('w', internal_callback)

    def get(self):
        """
        Retrieve the value of the dropdown

        :return: the value of the current variable
        """
        return self._var.get()

    def set(self, value):
        """
        Set the value of the dropdown

        :param value: a string representing the
        :return: None
        """
        self._var.set(value)
