from threading import RLock


class Singleton:
    "有时会用到 单例模式"
    _instance_lock = RLock()
    _singleton_instance = None

    def __init__(self):
        with self._instance_lock:
            if self._singleton_instance is not None:
                raise Exception("Cannot instantiate singleton more than once.")

    @classmethod
    def singleton(cls):
        with cls._instance_lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = cls()
        return cls._singleton_instance

    @classmethod
    def reset_singleton(cls):
        with cls._instance_lock:
            if cls._singleton_instance is not None:
                del cls._singleton_instance
            cls._singleton_instance = None
