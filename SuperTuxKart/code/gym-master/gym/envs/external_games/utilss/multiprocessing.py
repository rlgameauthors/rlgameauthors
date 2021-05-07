import threading
import time
import redis
import pickle


class RedisHandler:
    def __init__(self, hostname, namespace="gym_rh_"):
        self._redis = redis.Redis(hostname)
        self._namespace = namespace
        self._handlers = []
    
    def save(self, name, obj):
        pickled = pickle.dumps(obj)
        self._redis.set(f"{self._namespace}{name}", pickled)
    
    def load(self, name):
        pickled = self._redis.get(f"{self._namespace}{name}")
        if pickled is None:
            return None
        
        result = pickle.loads(pickled)
        return result
    
    def call_method(self, name, **kwargs):
        params_id = f"mparams_{name}"
        self.save(params_id, kwargs)
        
        method_id = f"mcall_{name}"
        self.save(method_id, True)
        
        while self.load(method_id) is True:
            pass
    
    def set_method_responder(self, name, action):
        thread = threading.Thread(target=lambda: self._method_handler(name, action))
        thread.start()
        self._handlers.append(thread)
    
    def clear_method_call(self, name):
        params_id = f"mparams_{name}"
        method_id = f"mcall_{name}"
        self.save(params_id, None)
        self.save(method_id, None)
        
    def check_method_call(self, name, action, wait=False):
        params_id = f"mparams_{name}"
        method_id = f"mcall_{name}"
        
        if wait:
            self.clear_method_call(name)
            while not self.load(method_id):
                time.sleep(0.0001)
        else:
            if not self.load(method_id):
                return False
            
        params = self.load(params_id)
        
        if params is not None:
            action(**params)
        else:
            action()
        
        self.save(method_id, False)
        return True
    
    def _method_handler(self, name, action):
        while True:
            self.check_method_call(name, action, wait=True)
