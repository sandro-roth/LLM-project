import time
import logging
from functools import wraps

def timeit(func):
    '''Decorator to measure execution time'''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        logger = logging.getLogger(__name__)
        logger.info(
            f"Function {func.__qualname__} took {total_time:.4f} seconds "
            f"with args={args} kwargs={kwargs}"
        )

        return result
    return wrapper