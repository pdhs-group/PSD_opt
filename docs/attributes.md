(attributes_overview)= 
# Attributes of the dPBE Class

## General Information

The following tables summarize the most important class attributes for performing PBE simulations. 
> Note: There are more! See the ``__init__( )`` method of ``pop.py`` 

All of them can be set directly during or manually after instancing, as shown in this example.
```python
# Setting the attributes directly during instancing
p = pop(1, NS=5, S=1.2)

# Setting the attributes after instancing
p = pop(1)
p.NS, p.S = 5, 1.2
```


## Model Parameters
| Attribute | Type | Default | Description |  
|---|---|---|---|
| ``dim`` | ``int`` | $-$ | Dimension of the PBE, i.e. number of materials. $\in\{1,2,3\}$ |
| ``disc`` | ``str`` | ``'geo'`` | Discretization scheme. $\in\{$``'geo'``,``'uni'``$\}$ |
| ``NS`` | ``int`` | 12 | Number of grid points |
| ``S`` | ``float`` | 2 | Spacing of geometric grid $V_i=s^{i}V_0$ |
| ``COLEVAL`` | ``int`` | 1 | Case for calculation of beta. 1 = orthokinetic, 2 = perikinetic |
| ``EFFEVAL`` | ``int`` | 2 | Case for calculation of alpha. 1 = Full calculation, 2 = Reduced model (only based on primary particle interactions) |
| ``USE_PSD`` | ``bool`` | ``True`` |  Define whether or not the PSD should be initializes (False = monodisperse primary particles) |
| ``SIZEEVAL`` | ``int`` | 2 |  Case for implementation of size dependency. 1 = No size dependency, 2 = Model from Soos2007 |
| ``X_SEL`` | ``float`` | 0.31 |  Size dependency parameter for Selomulya2003 / Soos2006 |
| ``Y_SEL`` | ``float`` | 1.06 |  Size dependency parameter for Selomulya2003 / Soos2006 |
| ``CORR_BETA`` | ``float`` | 25 | Correction Term for collision frequency [-] |


## Material Specific Parameters

| Attribute | Type | Default | Description | 
|---|---|---|---|
| ``R01`` | ``float`` | 2.9e-7 | Radius primary particle component 1 [m]  |
| ``R02`` | ``float`` | 2.9e-7 | Radius primary particle component 2 [m]  |
| ``R03`` | ``float`` | 2.9e-7 | Radius primary particle component 3 [m]  |
| ``DIST1`` | ``str`` | ``./data/PSD_data/PSD_x50_1.0E-6_r01_2.9E-7.npy`` | Absolute path to *.npy* file for PSD of material 1  |
| ``DIST2`` | ``str`` | ``./data/PSD_data/PSD_x50_1.0E-6_r01_2.9E-7.npy`` | Absolute path to *.npy* file for PSD of material 2  |
| ``DIST3`` | ``str`` | ``./data/PSD_data/PSD_x50_1.0E-6_r01_2.9E-7.npy`` | Absolute path to *.npy* file for PSD of material 3  |
| ``alpha_prim`` | ``array-like`` | np.ones(dim**2) | Collision efficiency array |

## Experimental Parameters

| Attribute | Type | Default | Description | 
|---|---|---|---|
| ``t_exp`` | ``float`` | 10 | Agglomeration time [min]  |
| ``c_mag_exp`` | ``float`` | 0.01 | Volume concentration of magnetic particles [Vol-%] |
| ``Psi_c1_exp`` | ``float`` | 1 | Concentration ratio component 1 (V_NM1/V_M) [-]  |
| ``Psi_c2_exp`` | ``float`` | 1 | Concentration ratio component 2 (V_NM2/V_M) [-]  |
| ``V0i`` | ``float`` | $-$ | Total volume concentration of component $i$ [m³/m³]. Calculated from ``c_mag_exp`` and ``Psi_c1_exp`` |
| ``N0i`` | ``float`` | $-$ | Total number concentration of primary particles component $i$ [1/m³]. Calculated from ``V0i`` and ``R0i`` |