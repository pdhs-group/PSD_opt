(quickstart)= 
# Quick Start

First of all, import the population class into your python script. Make sure that your interpreter is able to find the module (i.e. add the path or position the files accordingly)
```python
from pop import population as pop
```
Each calculation is performed as a stand-alone **population**. This is an *instance* of the population class and contains all relevant methods parameters as attributes. The only attribute that you must define during instancing is the dimension ``dim``. All other attributes can either be supplied via the ``**attr`` keyword or changed manually later. The following code creates two populations (class instances) ``p1`` and ``p2``. The first one is 1D the second one is 2D. For ``p1``, some attributes are adjusted during instancing and for ``p2`` some attributes are adjusted after instancing (both is valid).
> Note: For more information on the attributes see the [the overview of most important attributes](attributes_overview)
> During instancing, the ``__init__( )`` of the population class is executed (look there for more information)
```python
p1 = pop(1, NS=5, S=1.2)
p2 = pop(2)
p2.NS, p2.S = 15, 1.3
p2.alpha_prim = np.array([0.1, 0.2, 0.2, 1])
```

Currently, both instances only contain the initialized attributes and methods of the class. For calculation, the volume grid ``V`` (``R`` for radii), number density array ``N``, collision frequency array ``alpha_prim`` (optional), agglomeration efficiency array ``F_M`` and (not fully implemented) breakage rate array ``B_M`` need to be set up. This can either be done manually by calling each function individually or by simply calling ``full_init( )``.
> Note: ``alpha_prim`` is initialized with ones (fully destabilized). You can either define the $\alpha$ values statically (see code above) or estimate them from material data (not predictive and not recommended) by calling ``calc_alpha_prim( )``. If you want to use ``full_init( )`` with statically set ``alpha_prim`` set the parameter ``calc_alpha=False``.
```python
# Calling the methods individually
p1.calc_R()
p1.init_N()
# p1.calc_alpha_prim() # Not recommended
p1.calc_F_M()
p1.calc_B_M()

# Using the full_init method
p2.full_init(calc_alpha=False)
```

Let's solve the PBE with the default settings. Use ``solve_PBE( )``. It is advised to provide the time points at which the solution should be returned via the ``t_vec`` argument. The ODE is solved with [``scipy.solve_ivp( )``](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).
```python
# Calculate for 600 seconds and return the solution every 2 seconds.
t_vec_def = np.arange(0,600,2)
p1.solve_PBE(t_vec = t_vec_def)
p2.solve_PBE(t_vec = t_vec_def)
```

Congratulations, you solved the PBE for the provided parameters. The solution is stored inside ``p.N``, where the last index specifies the time index. To manually inspect this array (or any other attribute of ``p1`` and ``p2``), you can *"bring them into your workspace"* as shown in the following code block.
> Note: When using Spyder, ``n1`` and ``n2`` should now show up under the *Variable Explorer*.
```python
n1 = p1.N
print(n1)
n2 = p2.N
print(n[:,:,-1])
```

 You can also use some built-in methods to visualize the results:
```python
p2.visualize_distN_t()
p2.visualize_qQ_t()
p2.visualize_sumN_t()
```
