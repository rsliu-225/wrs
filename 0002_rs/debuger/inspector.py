class Error_info():
    def __init__(self, name=None, objmat=None, lftgrasppose=None, rgtgrasppose=None, lftarmjnts=None, rgtarmjnts=None,
                 lftjawwidth=None, rgtjawwidth=None):
        self.id = id(self)
        self.name = name
        self.objmat = objmat
        self.lftgrasppose = lftgrasppose
        self.rgtgrasppose = rgtgrasppose
        self.lftarmjnts = lftarmjnts
        self.rgtarmjnts = rgtarmjnts
        self.lftjawwidth = lftjawwidth
        self.rgtjawwidth = rgtjawwidth


class Inspector:
    def __init__(self):
        self.error_detail = {}
        # {
        # 1. errortype:
        # 2. error
        # }
        self.error_type = {}
        # {error_name: parent error} => if parent error is None: top level error

    def register_error_type(self, error_type, parent_error=None):
        if error_type not in self.error_type:

            if parent_error is None or parent_error in self.error_type:
                self.error_type[error_type] = parent_error
            else:
                raise Exception("parent error is not exist")
        else:
            raise Exception("Error type has already existed !")

    def add_error(self, error_type, error_info, parent_error=None):
        """

        :param error_type:
        :param error_info:  type of Error info
        :param parent_error: the parent_error, if error_type is in the
        :return:
        """
        if not isinstance(error_info, Error_info):
            raise Exception(f"error info should be instance of {type(Error_info)}")
        if error_type not in self.error_type:
            self.register_error_type(error_type=error_type, parent_error=parent_error)

        error = self.error_detail.get(error_type, [])
        if len(error) > 0:
            error.append(error_info)
        else:
            self.error_detail.setdefault(error_type, [error_info])

    def get_parent_error(self):
        return [error_name for error_name in self.error_type.keys() if self.error_type[error_name] is not None]

    def get_child_error(self, parent_error):
        return [error_name for error_name, error_parent in self.error_type.items() if error_parent == parent_error]

    def get_error(self):
        return self.error_detail.keys()

    def get_error_info(self, error_name):
        return self.error_detail.get(error_name, [])

    def get_error_info_id(self, error_name):
        error_info = self.get_error_info(error_name)
        return {str(error.id): error for error in error_info}

    def get_error_info_name_id(self, error_name):
        error_info = self.get_error_info(error_name)
        return [f"{error.id}:{error.name}" for error in error_info]


if __name__ == '__main__':
    import tkinter as tk
    from gui.core.inspector import InspectorPanel

    root = tk.Tk()
    inspector = Inspector()
    inspector.add_error("error", Error_info(name=1))
    child = InspectorPanel(root, inspector)
    root.mainloop()
