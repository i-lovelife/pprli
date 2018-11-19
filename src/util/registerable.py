from collections import defaultdict
class Registerable:
    _registry = defaultdict(dict)
    
    @classmethod
    def register(cls, name):
        cls_registed = Registerable._registry[cls]
        def add_to_registry(subclass):
            if name in cls_registed:
                raise ValueError(f'{name}already registed in {cls.__name__}, \
                        old:{cls_registed[name].__name__} new:{subclass.__name__}')
            cls_registed[name] = subclass
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