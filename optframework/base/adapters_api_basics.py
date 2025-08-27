# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:04:13 2025

@author: px2030
"""
from typing import Any, Callable, Set

# Type alias: a custom setter function receives (impl, value)
SetterFn = Callable[[Any, Any], None]


class WriteThroughAdapter:
    """
    A base adapter that synchronizes ("writes through") attributes
    between the adapter itself and an underlying solver instance (`impl`).

    Main behavior:
    --------------
    - When setting an attribute (e.g. setattr(adapter, name, value)):
        1. The value is always stored in the adapter itself.
        2. If allowed, the value is also propagated into the underlying solver (`impl`).

    - When getting an attribute (e.g. getattr(adapter, name)):
        1. If the attribute exists in the adapter, return it.
        2. Otherwise, try to look it up in the solver (`impl`).
        3. Internal fields (like `_map`, `_setters`) are never proxied.

    Special controls:
    -----------------
    - `_map`:   maps adapter attribute names -> impl attribute names.
    - `_setters`: custom setter functions for attributes requiring special logic.
    - `_skip`:   attributes that should only exist on the adapter, not written through.
    """

    # Reserved internal names (not to be proxied to the solver)
    _INTERNAL: Set[str] = {"impl", "_map", "_setters", "_skip"}

    def __init__(self, impl: Any):
        """
        Initialize the adapter with a solver instance.

        Parameters
        ----------
        impl : Any
            The underlying solver object (e.g., DPBESolver).
        """
        # Use super().__setattr__ to bypass our overridden __setattr__
        super().__setattr__("impl", impl)
        super().__setattr__("_map", {})       # attribute name mapping
        super().__setattr__("_setters", {})   # custom setters
        super().__setattr__("_skip", set())   # attributes skipped from impl sync

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Intercept attribute assignment.
    
        Logic:
        ------
        1. If it's an internal field, only update the adapter.
        2. Otherwise, always store the value on the adapter itself.
        3. If the attribute is marked as skipped -> do nothing further.
        4. If a custom setter exists -> call it with (impl, value).
        5. Otherwise, map the name and set it on the solver if possible.
           - If the solver does not already have this attribute,
             still set it, but print a warning.
        """
        if name in self._INTERNAL:
            super().__setattr__(name, value)
            return
    
        # Always store value locally in the adapter
        super().__setattr__(name, value)
    
        # Use custom setter if defined
        if name in self._setters:
            self._setters[name](self.impl, value)
            return
    
        # Skip write-through if marked
        if name in self._skip:
            return
    
        # Default: map the name and set on impl
        target = self._map.get(name, name)
        if hasattr(self.impl, target):
            setattr(self.impl, target, value)
        else:
            # print(f"[Warning] Adapter added new attribute '{target}' to impl "
            #       f"of type {type(self.impl).__name__}.")
            setattr(self.impl, target, value)

    def __getattr__(self, name: str) -> Any:
        """
        Fallback attribute lookup (called only if normal lookup fails).

        Logic:
        ------
        1. Do not proxy internal fields (raise AttributeError).
        2. Map the attribute name (if needed).
        3. If the solver (`impl`) has the attribute, return it.
        4. Otherwise, raise AttributeError.
        """
        if name in self._INTERNAL:
            raise AttributeError(name)

        # Resolve mapped name if applicable
        target = self._map.get(name, name)
        if hasattr(self.impl, target):
            return getattr(self.impl, target)

        raise AttributeError(name)
