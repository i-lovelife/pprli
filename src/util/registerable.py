from collections import defaultdict
class Registerable:
    _registry = defaultdict(dict)
    
    def __init__(self, **args):
        if args != {}:
            raise ValueError(f'{args} is not None for init {type(self).__name__}')
    @classmethod
    def register(cls, name):
        cls_registed = Registerable._registry[cls]
        def add_to_registry(subclass):
            if name in cls_registed:
                raise ValueError(f'{name}already registed in {cls.__name__}, \
                        old:{cls_registed[name].__name__} new:{subclass.__name__}')
            cls_registed[name] = subclass
            return subclass
        return add_to_registry
    @classmethod
    def by_name(cls, name):
        cls_registed = Registerable._registry[cls]
        if name not in cls_registed:
            raise ValueError(f'{name} not registed in {cls.__name__}')
        return cls_registed[name]
    @classmethod
    def list_all(cls):
        cls_registed = Registerable._registry[cls]
        for key, value in cls_registed.items():
            print(f'{key}:{value.__name__}')

    @classmethod
    def from_hp(cls, hp):
        if "type" not in hp:
            raise ValueError(f'type not in hp:{hp}')
        type = hp.pop("type")
        return cls.by_name(type)(**hp)
