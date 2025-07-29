import threading
import time


class RateLimiter:
    def __init__(self, rate_per_second):
        self.rate = rate_per_second
        self.lock = threading.Lock()
        self.last_time = time.time()

    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_time
            wait_time = max(0, 1 / self.rate - elapsed)
            if wait_time > 0:
                time.sleep(wait_time + 0.5)
            self.last_time = time.time()
