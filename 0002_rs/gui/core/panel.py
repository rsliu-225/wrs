from gui.core.widgets.base import GuiFrame


class Panel(GuiFrame):
    def __init__(self, root, title="undefined", structuretable=None):
        """

        :param root:
        :param title:
        :param structuretable:
        {
            {
            "title":,
            "showtitle":bool
            "component":
            "component_para":dict
            }
            {
            "title":,
            "component"
            }

        }
        """
        GuiFrame.__init__(self, root=root, topmost=True, close_hidden=True)
        self.set_title(title)
        if structuretable is not None:
            for structure in structuretable:
                if "showtitle" not in structure:
                    self.add_title(structure["title"])
                else:
                    pass

                self.widgets[structure["title"]] = structure["component"](self, **structure["component_para"])
                if self.widgets[structure["title"]]:
                    self.widgets[structure["title"]].grid(row=self.irow)

        self.set_size(300, len(self.widgets) * 120)


if __name__ == '__main__':
    import tkinter as tk

    root = tk.Tk()
    child = Panel(root, structuretable={
        {
            "title": "hi",
            "component": None,
        },
        {
            "title": "hi",
            "component": None,
        },
        {
            "title": "hi",
            "component": None,
        }

    })
    root.mainloop()
