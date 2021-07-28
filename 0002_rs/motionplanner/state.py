from gui.core.widgets.base import GuiFrame
from config import STATE


def initilizestate(base, taskMgr, **kwargs):
    STATE['base'] = base
    STATE['taskMgr'] = taskMgr
    STATE['SimpleTaskMgr'] = SimpleTaskMgr()
    for key, value in kwargs.items():
        STATE.setdefault(key, value)


def registeration(components):
    STATE['base'].startTk()
    STATE['base'].tkRoot.withdraw()
    STATE['taskMgr'].doMethodLater(.1, STATE['SimpleTaskMgr'].loop, "SimpleTaskMgr")
    STATE['components'] = []
    for com in components:
        para = []
        if isinstance(com, list):
            com, para = com[0], com[1:]
        if issubclass(com, GuiFrame):
            com_instance = com(STATE['base'].tkRoot, *para)
        else:
            com_instance = com()
        com_instance.registeration()
        com_instance.add()
        STATE['components'].append(com_instance)


class SimpleTaskMgr:
    def __init__(self):
        self.__tasktable = []
        self.__commonvariable = {}

    def add(self, task, args=[], time=0):
        """

        :param task:
        :param args:
        :param time: time equal to 0: execute once
        :return:
        """
        self.__tasktable.append({"task": task, "args": args, "time": time, "counter": time})

    # def register(self,name,val):
    #     self.__commonvariable

    def loop(self, task):
        removelist = []
        for task_dict in self.__tasktable:
            task_dict["counter"] -= 0.1
            if task_dict["counter"] <= 0:
                task_dict["task"](*task_dict["args"]) if isinstance(task_dict["args"], list) else task_dict["task"](
                    **task_dict["args"])
                task_dict["counter"] = task_dict["time"]
                if task_dict["time"] == 0:
                    removelist.append(task_dict)
        if len(removelist) > 0:
            [self.__tasktable.pop(self.__tasktable.index(item)) for item in removelist]
        return task.again
