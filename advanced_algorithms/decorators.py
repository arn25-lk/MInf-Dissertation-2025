# decorators.py
import time
import queue
import os
from functools import wraps
from dotenv import load_dotenv


load_dotenv()
# Global queue for data collection
data_queue = queue.Queue()


# Check if we are in testing mode (default False)
TESTING_MODE = os.getenv("TESTING_MODE", "false").lower() == "true"
RUNTIME = os.getenv("RUNTIME", "false").lower() == "true"
ALG_RUNTIME = os.getenv("ALG_RUNTIME", "false").lower() == "true"
GRAPH_DATA = os.getenv("GRAPH_DATA", "false").lower() == "true"
FREQUENCY = os.getenv("FREQUENCY", "false" ).lower() == "true"



def runtime(decorator):
    """Apply a decorator only if TESTING_MODE is enabled."""
    def wrapper(func):
        if TESTING_MODE and RUNTIME:
            return decorator(func)
        return func
    return wrapper


def alg_runtime(decorator):
    """Apply a decorator only if TESTING_MODE is enabled."""
    def wrapper(func):
        if TESTING_MODE and ALG_RUNTIME:
            return decorator(func)
        return func
    return wrapper


def frequency(decorator):
    """Apply a decorator only if TESTING_MODE is enabled."""
    def wrapper(func):
        if TESTING_MODE and FREQUENCY:
            return decorator(func)
        return func
    return wrapper



def graph_data(decorator):
    """Apply a decorator only if TESTING_MODE is enabled."""
    def wrapper(func):
        if TESTING_MODE and GRAPH_DATA:
            return decorator(func)
        return func
    return wrapper



@graph_data
def collect_graphs_on_call(func):
    """Decorator to collect graph data."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        inst_obj = args[0]
        data = {
            "function": func.__name__,
            "args": args,
            "kwargs": kwargs,
            "envy_graph": inst_obj.gen.get_envy_graph()
        }
        data_queue.put(data)
        return func(*args, **kwargs)
    return wrapper




@frequency
def collect_frequency_on_call(func):
    """Decorator to collect graph data."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if data_queue.empty():
            data = {
            }
        else: 
            data : dict = data_queue.get()
        if func.__name__ in data:
            data[func.__name__] += 1 

            data_queue.put(data)
        else:
            data[func.__name__] = 1
            data_queue.put(data)
        
        return func(*args, **kwargs)
    return wrapper


@runtime
def collect_timestamps_on_call(func):
    """Decorator to collect timing data before and after function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        data = {
            "function": func.__qualname__,
            "start_time": start_time,
            "end_time": end_time,
            "exec_time": end_time - start_time,
        }
        data_queue.put(data)
        return result
    return wrapper


@alg_runtime
def collect_alg_timestamps_on_call(func):
    """Decorator to collect timing data before and after function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        data = {
            "function": func.__qualname__,
            "start_time": start_time,
            "end_time": end_time,
            "exec_time": end_time - start_time,
        }
        data_queue.put(data)
        return result
    return wrapper



