import tkinter as tk
import tkinter.filedialog as tkfiledialog
from .widget import Widget


class FileBrowser(Widget):
    """
    Open the file
    """

    def __init__(self, parent, title: str = "Select File", initialdir: str = "/",
                 file_parameters: tuple[tuple[str]] = (("jpeg files", "*.jpg"), ("all files", "*.*"))
                 , callback: callable = None):
        super().__init__(parent)

        self.filebrowser = tkfiledialog.askopenfilename(initialdir=initialdir, title="Select file",
                                                        filetypes=file_parameters)
