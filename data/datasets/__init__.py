
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt import MSMT17_V1
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17_V1,
    'dukemtmc': DukeMTMCreID
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
