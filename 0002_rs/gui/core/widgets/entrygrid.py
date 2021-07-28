import tkinter.ttk as ttk
from gui.core.widgets.grid import _Grid


class EntryGrid(_Grid):
    """
    Add a spreadsheet-like grid of entry widgets.

    :param parent: the tk parent element of this frame
    :param num_of_columns: the number of columns contained of the grid
    :param headers: a list containing the names of the column headers
    """

    def __init__(self, parent,
                 num_of_columns: int, headers: list = None,
                 **options):
        super().__init__(parent, num_of_columns, headers, **options)

    def add_row(self, data: list = None):
        """
        Add a row of data to the current widget, add a <Tab> \
        binding to the last element of the last row, and set \
        the focus at the beginning of the next row.

        :param data: a row of data
        :return: None
        """
        # validation
        if self.headers and data:
            if len(self.headers) != len(data):
                raise ValueError

        offset = 0 if not self.headers else 1
        row = list()

        if data:
            for i, element in enumerate(data):
                contents = '' if element is None else str(element)
                entry = ttk.Entry(self)
                entry.insert(0, contents)
                entry.grid(row=len(self._rows) + offset,
                           column=i,
                           sticky='E,W')
                row.append(entry)
        else:
            for i in range(self.num_of_columns):
                entry = ttk.Entry(self)
                entry.grid(row=len(self._rows) + offset,
                           column=i,
                           sticky='E,W')
                row.append(entry)

        self._rows.append(row)

        # clear all bindings
        for row in self._rows:
            for widget in row:
                widget.unbind('<Tab>')

        def add(e):
            self.add_row()

        last_entry = self._rows[-1][-1]
        last_entry.bind('<Tab>', add)

        e = self._rows[-1][0]
        e.focus_set()

        self._redraw()

    def _read_as_dict(self):
        """
        Read the data contained in all entries as a list of
        dictionaries with the headers as the dictionary keys

        :return: list of dicts containing all tabular data
        """
        data = list()
        for row in self._rows:
            row_data = {}
            for i, header in enumerate(self.headers):
                row_data[header.cget('text')] = row[i].get()

            data.append(row_data)

        return data

    def _read_as_table(self):
        """
        Read the data contained in all entries as a list of
        lists containing all of the data

        :return: list of dicts containing all tabular data
        """
        rows = list()

        for row in self._rows:
            rows.append([row[i].get() for i in range(self.num_of_columns)])

        return rows

    def read(self, as_dicts=True):
        """
        Read the data from the entry fields

        :param as_dicts: True if list of dicts required, else False
        :return: entries as a dict or table
        """
        if as_dicts:
            return self._read_as_dict()
        else:
            return self._read_as_table()
