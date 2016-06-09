class EasyDict(dict):
    def __init__(self, **kwargs):
        pass

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

