# agggenerator

## Module Introduction

`agggenerator` provides reusable aggregate generation utilities for the monorepo PBE workflow. It focuses on building synthetic particle/agglomerate structures and preparing PSD-related inputs for downstream simulation modules.

- Provides core generators in `src/agggenerator` for 2D MPTSA lattice growth, material assignment, and PSD-driven growth.
- Includes preprocessing scripts for CPS PSD data conversion/packaging and pool-building scripts for offline sampling workflows.
- Contains lightweight test/demo scripts under `tests` to validate generation and LMC integration behavior.
