# Advanced Guide: Coupling Custom Solvers with the Optimizer

In this framework, the coupling between solvers and the optimizer is handled through the **Adapters API**.  

The purpose of this design is to **decouple optimizer logic from solver-specific calls**.  
When adding a new solver, you do not need to modify the optimizer itself — you only provide an adapter that translates between the solver’s API and the optimizer’s requirements.

---

## 1. Adapters API Modules

The Adapters API is defined across three modules:

- **`optframework.base.adapters_api_basics`**  
- **`optframework.base.adapters_api`**  
- **`optframework.base.custom_adapter`**

Among them:

- **`custom_adapter`** → contains solver-specific adapter implementations.  
  - Users must implement this to integrate their custom solver.  
  - Example: **{py:class}`DPBESolver Adapter <optframework.base.dpbe_adapter.DPBEAdapter>`**

- **`adapters_api_basics`** → provides the base class `WriteThroughAdapter`.  
  - This is a *write-through* mechanism: when adapter attributes are updated, the corresponding solver attributes are updated automatically.  
  - The mapping can be:  
    - One-to-one (same attribute names)  
    - Mapped (different attribute names)  
    - Processed (via a function before assignment)  
  - Useful when optimizer and solver use attributes with similar meaning but different names or formats.  
  - See: **{py:class}`WriteThroughAdapter <optframework.base.adapters_api_basics.WriteThroughAdapter>`**

- **`adapters_api`** → defines the protocol for solver adapters.  
  - Specifies which methods must be implemented, since they are called during optimization.  
  - See: **{py:class}`SolverProtocol <optframework.base.adapters_api.SolverProtocol>`**

---

## 2. Key Points When Defining a Custom Adapter

1. **Method call order**  
   - Each required method has a well-defined calling stage (documented in `SolverProtocol`).  
   - Ensure that later methods have access to all variables prepared by earlier methods.

2. **Access to Optimizer Core**  
   - During initialization, a `CustomAdapter` receives a reference to the optimizer’s **Core instance** (`opt`).  
   - This means the adapter can directly access optimizer attributes and methods.  
   - The `CustomAdapter` instance is also stored as an attribute inside the optimizer Core.  
   - If multiple adapters/solvers are instantiated in one optimization run, they can call each other directly.  
   - ⚠️ This may require structural adjustments in the optimizer, such as when using the **Multi computation structure** (where three solvers are instantiated simultaneously with data exchange between them).

3. **Internal attributes of CustomAdapter**  
   When initialized, the adapter creates four key internal attributes:

   - **`impl`** → pointer to the coupled solver instance.  
   - **`_map`** → mapping rules between adapter attributes and solver attributes.  
     - Example:  
       ```python
       self._map.update({"grid_x": "V1"})
       ```
       Here, assigning `adapter.grid_x = value` will actually update the solver’s `V1` attribute.  
   - **`_skip`** → attributes that should be stored only in the adapter and not passed to the solver.  
   - **`_setters`** → attributes that require processing by a custom function before being passed to the solver.  
     - Example usage:  
       ```python
       self._setters["my_attr"] = custom_func
       ```
       - When `adapter.my_attr` is updated, the value is sent to `custom_func`.  
       - Inside `custom_func`, you must explicitly assign the processed value to the solver — the function should **not** return a value to be auto-assigned.  

   ⚠️ Note: For `_setters`, the adapter still keeps its own copy of the **pre-processed value**.

4. **Attribute persistence**  
   - Regardless of whether attributes are passed to the solver, the adapter always keeps a copy of them internally.  
   - This ensures consistency for inspection and debugging.  
   - For `_setters`, the adapter stores the **raw (unprocessed)** value internally, while the solver receives the processed one.

---

## 3. Summary

- Adapters serve as the **bridge** between optimizers and solvers.  
- They preserve the optimizer’s structure while allowing flexible integration of different solvers.  
- Key mechanisms:  
  - **Write-through updates** via `WriteThroughAdapter`  
  - **Protocol enforcement** via `SolverProtocol`  
  - **Custom mapping, skipping, and preprocessing** through `_map`, `_skip`, and `_setters`  

By following this pattern, users can integrate their **own solvers** into the optimization framework with minimal changes to the optimizer logic itself.


```{toctree}
:caption: 'Advanced Guide:'
:maxdepth: 1

