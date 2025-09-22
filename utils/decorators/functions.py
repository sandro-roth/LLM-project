import time
import logging
from functools import wraps

def timeit(func):
    '''Decorator to measure execution time'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            total_time = time.perf_counter() - start_time
            logger = logging.getLogger()
            logger.info("Function %s took %.4f s",
                        func.__qualname__, total_time)
    return wrapper