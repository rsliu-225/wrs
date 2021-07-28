import tkinter as tk
from .widget import Widget


class MenuBar(Widget):
    """
     dict = {
     bar_name: {
        command: callable
        child: {
                bar_name: {}

                }
     }
     }
    """

    def __init__(self, parent, menu_options: dict):
        super().__init__(parent)

        def generate_menubar(root, options, isroot=False):
            menu_bar = tk.Menu(root)
            for menu_name, menu_val in options.items():
                if menu_val.get('child', False):
                    menu_bar.add_cascade(label=menu_name, menu=generate_menubar(menu_bar, options=menu_val['child']))
                else:
                    menu_bar.add_command(label=menu_name, command=menu_val['command'])
            return menu_bar

        self.menu_bar = generate_menubar(self, menu_options, isroot=True)
        # self.config(menu =self.menu_bar)
