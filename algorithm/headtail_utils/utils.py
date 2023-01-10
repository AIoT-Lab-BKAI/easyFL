import importlib

def initialize(dataset, architecture, modelname):
    if modelname not in ["ClientModel", "ClientHead", "ClientTail", "ServerTail"]:
        raise Exception("Do not support model name:", modelname)
    
    path = '%s.%s' % (dataset, architecture)
    model = getattr(importlib.import_module(path), modelname)
    return model()
    