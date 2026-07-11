from glo_cache import Cache

class LocalCache(Cache):

    def __init__(self):
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)
    
    def set(self, key, value):
        self.cache[key] = value

    def delete(self, key):
        if key in self.cache:
            del self.cache[key]