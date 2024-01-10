import sys
import numbers

class Named(type):
    def __str__(self):
        return self.__name__
    def __repr__(self):
        return self.__name__

def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

def flatten_dict(d:dict):
    """ Flattens a dictionary, ignoring outer keys. Only
        numbers and strings allowed, others will be converted
        to a string. """
    out = {}
    for k,v in d.items():
        if isinstance(v,dict):
            out.update(flatten_dict(v))
        elif isinstance(v,(numbers.Number,str,bytes)):
            out[k] = v
        else:
            out[k] = str(v)
    return out