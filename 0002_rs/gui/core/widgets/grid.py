import tkinter.ttk as ttk
import tkinter as tk


class _Grid(ttk.Frame):
    padding = 3

    """
    Creates a grid of widgets (intended to be subclassed).

    :param parent: the tk parent element of this frame
    :param num_of_columns: the number of columns contained of the grid
    :param headers: a list containing the names of the column headers
    """

    def __init__(self, parent, num_of_columns: int, headers: list = None,
                 **options):
        self._parent = parent
        super().__init__(self._parent, padding=3, borderwidth=2,
                         **options)
        self.grid()

        self.headers = list()
        self._rows = list()
        self.num_of_columns = num_of_columns

        # do some validation
        if headers:
            if len(headers) != num_of_columns:
                raise ValueError

            for i, element in enumerate(headers):
                label = ttk.Label(self, text=str(element), relief=tk.GROOVE,
                                  padding=self.padding)
                label.grid(row=0, column=i, sticky='E,W')
                self.headers.append(label)

    def add_row(self, data: list):
        """
        Adds a row of data based on the entered data

        :param data: row of data as a list
        :return: None
        """
        raise NotImplementedError

    def _redraw(self):
        """
        Forgets the current layout and redraws with the most recent information

        :return: None
        """
        for row in self._rows:
            for widget in row:
                widget.grid_forget()

        offset = 0 if not self.headers else 1
        for i, row in enumerate(self._rows):
            for j, widget in enumerate(row):
                widget.grid(row=i + offset, column=j)

    def remove_row(self, row_number: int = -1):
        """
        Removes a specified row of data

        :param row_number: the row to remove (defaults to the last row)
        :return: None
        """
        if len(self._rows) == 0:
            return

        row = self._rows.pop(row_number)
        for widget in row:
            widget.destroy()

    def clear(self):
        """
        Removes all elements of the grid

        :return: None
        """
        for i in range(len(self._rows)):
            self.remove_row(0)
