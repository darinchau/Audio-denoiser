from typing import TypeVar
import numpy as np

copyable = TypeVar("copyable")
def copy(obj: copyable) -> copyable:
    """Makes a deep copy. 
    If the parameter is a list/mutable iterable, returns the recursive deep copy
    If the parameter is a user-defined class, enable deep copy by defining a dunder copy method"""
    if isinstance(obj, int | float | str):
        return obj
    
    if isinstance(obj, list):
        return [copy(x) for x in obj] #type: ignore
    
    if isinstance(obj, tuple):
        return tuple(copy(x) for x in obj) #type: ignore
    
    if isinstance(obj, dict):
        return {copy(k): copy(v) for k, v in obj.items()} #type: ignore
    
    if isinstance(obj, np.ndarray):
        return np.array(obj, dtype = obj.dtype)
    
    if "__copy__" in dir(obj):
        return obj.__copy__() #type: ignore
    
    raise TypeError(f"Object of type {type(obj).__name__} is not copyable!")
