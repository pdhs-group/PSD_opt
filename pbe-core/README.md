# pbe-core

`pbe-core` provides reusable core components for Population Balance Equation (PBE) solvers.

- Includes JIT-accelerated numerical kernels under `src/pbe_core/func` for agglomeration, breakage, PBM, and extruder RHS evaluations.
- Provides base solver/post-processing utilities under `src/pbe_core/base`.
- Contains plotting helpers under `src/pbe_core/plotter` for standardized scientific visualization.
