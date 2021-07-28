import tkinter.ttk as ttk
import tkinter as tk


class KeyValueEntry(ttk.Frame):
    """
    Creates a key-value input/output frame.

    :param parent: the parent frame
    :param keys: the keys represented
    :param defaults: default values for each key
    :param unit_labels: unit labels for each key (to the right of the value)
    :param enables: True/False for each key
    :param title: The title of the block
    :param on_change_callback: a function callback when any element is changed
    :param options: frame tk options
    """

    def __init__(self, parent, keys: list, defaults: list = None,
                 unit_labels: list = None, enables: list = None,
                 title: str = None, on_change_callback: callable = None,
                 load=None,
                 **options):
        self._parent = parent
        super().__init__(self._parent, borderwidth=2,
                         padding=5, **options)

        # some checks before proceeding
        if defaults:
            if len(keys) != len(defaults):
                raise ValueError('unit_labels length does not '
                                 'match keys length')
        if unit_labels:
            if len(keys) != len(unit_labels):
                raise ValueError('unit_labels length does not '
                                 'match keys length')
        if enables:
            if len(keys) != len(enables):
                raise ValueError('enables length does not '
                                 'match keys length')

        self.keys = []
        self.values = []
        self.defaults = []
        self.unit_labels = []
        self.enables = []
        self.callback = on_change_callback

        if title is not None:
            self.title = ttk.Label(self, text=title)
            self.title.grid(row=0, column=0, columnspan=3)
        else:
            self.title = None

        for i in range(len(keys)):
            self.add_row(
                key=keys[i],
                default=defaults[i] if defaults else None,
                unit_label=unit_labels[i] if unit_labels else None,
                enable=enables[i] if enables else None
            )

        if load is not None:
            self.load({keys[i]: load[i] for i in range(len(keys))})

    def add_row(self, key: str, default: str = None,
                unit_label: str = None, enable: bool = None):
        """
        Add a single row and re-draw as necessary

        :param key: the name and dict accessor
        :param default: the default value
        :param unit_label: the label that should be \
        applied at the right of the entry
        :param enable: the 'enabled' state (defaults to True)
        :return:
        """
        self.keys.append(ttk.Label(self, text=key))

        self.defaults.append(default)
        self.unit_labels.append(
            ttk.Label(self, text=unit_label if unit_label else '')
        )
        self.enables.append(enable)
        self.values.append(ttk.Entry(self))

        row_offset = 1 if self.title is not None else 0

        for i in range(len(self.keys)):
            self.keys[i].grid_forget()

            self.keys[i].grid(row=row_offset, column=0, sticky='e')
            self.values[i].grid(row=row_offset, column=1)

            if self.unit_labels[i]:
                self.unit_labels[i].grid(row=row_offset, column=3, sticky='w')

            if self.defaults[i]:
                self.values[i].config(state=tk.NORMAL)
                self.values[i].delete(0, tk.END)
                self.values[i].insert(0, self.defaults[i])

            if self.enables[i] in [True, None]:
                self.values[i].config(state=tk.NORMAL)
            elif self.enables[i] is False:
                self.values[i].config(state=tk.DISABLED)

            row_offset += 1

            # strip <Return> and <Tab> bindings, add callbacks to all entries
            self.values[i].unbind('<Return>')
            self.values[i].unbind('<Tab>')

            if self.callback is not None:
                def callback(event):
                    self.callback()

                self.values[i].bind('<Return>', callback)
                self.values[i].bind('<Tab>', callback)

    def reset(self):
        """
        Clears all entries.

        :return: None
        """
        for i in range(len(self.values)):
            self.values[i].delete(0, tk.END)

            if self.defaults[i] is not None:
                self.values[i].insert(0, self.defaults[i])

    def change_enables(self, enables_list: list):
        """
        Enable/disable inputs.

        :param enables_list: list containing enables for each key
        :return: None
        """
        for i, entry in enumerate(self.values):
            if enables_list[i]:
                entry.config(state=tk.NORMAL)
            else:
                entry.config(state=tk.DISABLED)

    def load(self, data: dict):
        """
        Load values into the key/values via dict.

        :param data: dict containing the key/values that should be inserted
        :return: None
        """
        for i, label in enumerate(self.keys):
            key = label.cget('text')
            if key in data.keys():
                entry_was_enabled = \
                    str(self.values[i].cget('state')) == 'normal'
                if not entry_was_enabled:
                    self.values[i].config(state='normal')

                self.values[i].delete(0, tk.END)
                self.values[i].insert(0, str(data[key]))

                if not entry_was_enabled:
                    self.values[i].config(state='disabled')

    def get(self):
        """
        Retrieve the GUI elements for program use.

        :return: a dictionary containing all \
        of the data from the key/value entries
        """
        data = dict()
        for label, entry in zip(self.keys, self.values):
            data[label.cget('text')] = entry.get()

        return data
