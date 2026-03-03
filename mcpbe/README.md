# mcpbe

`mcpbe` contains Monte Carlo Population Balance Equation solvers for PSD simulation in this monorepo.

- Provides the standard solver implementation under `src/mcpbe` with aggregation, breakage, and post-processing components.
- Provides the weighted solver variant under `src/wmcpbe` with weighted particle handling and reconstruction utilities.
- Integrates with shared kernels/utilities from `pbe-core` and optional LMC/energy-model adapters used in breakage workflows.
