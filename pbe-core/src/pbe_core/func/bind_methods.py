# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:54:38 2025

@author: px2030
"""
import inspect
import importlib

# Function to bind all methods from a module to a class    
def bind_methods_from_module(cls, module_name):
    """
    Bind all functions from a specified module to the given class.

    Parameters
    ----------
    cls : type
        The class to which the methods will be bound.
    module_name : str
        The name of the module from which the methods will be imported.
    """
    # Import the specified module
    module = importlib.import_module(module_name)

    # # Iterate over all functions in the module and bind them to the class
    for name, func in inspect.getmembers(module, inspect.isfunction):
        # Bind functions statically to classes, with function names as method names
        setattr(cls, name, func)

# Function to bind selected methods from a module to a class        
def bind_selected_methods_from_module(cls, module_name, methods_to_bind):
    """
    Bind selected functions from a specified module to the given class.
    
    Parameters
    ----------
    cls : type
        The class to which the methods will be bound.
    module_name : str
        The name of the module from which the methods will be imported.
    methods_to_bind : list of str
        A list of method names to bind to the class.
    """
    # Import the specified module
    module = importlib.import_module(module_name)
    
    # Bind only the selected methods to the class
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name in methods_to_bind:
            setattr(cls, name, func)
    
# Function to unbind methods from a class
def unbind_methods_from_class(cls, methods_to_remove):
    """
    Unbind specific methods from the given class.
    
    Parameters
    ----------
    cls : type
        The class from which the methods will be unbound.
    methods_to_remove : list of str
        A list of method names to remove from the class.
    """
    for method_name in methods_to_remove:
        if hasattr(cls, method_name):
            delattr(cls, method_name)  