import config

STATE = config.STATE


def replaceat(strz, index, val):
    return strz[:index] + str(val) + strz[index + 1:]


def setkeys(k, v):
    STATE['base'].inputmgr.keymap[k] = v


def addkey(keys):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if key in STATE['base'].inputmgr.keymap: continue
        STATE['base'].inputmgr.keymap[key] = False
        STATE['base'].inputmgr.accept(key, setkeys, [key, True])
        STATE['base'].inputmgr.accept(key + '-up', setkeys, [key, False])


def addtask(task, args=None, timestep=0.1):
    if args is not None:
        STATE['taskMgr'].doMethodLater(timestep, task, task.__code__.co_name,
                                       extraArgs=args,
                                       appendTask=True)
    else:
        STATE['taskMgr'].doMethodLater(timestep, task, task.__code__.co_name)
