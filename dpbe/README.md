# dpbe
## Module Introduction

This project mainly consists of three fundamental modules:

`dpbe.DPBESolver`**: The current dPBE solver is capable of calculating particle aggregation and breakage processes, supporting both single and dual-component systems.
   - **Uni-grid**: Corresponds to the general discretization method, suitable for standard verification but may contain bugs.  
   - **Geo-grid**: Uses the Cell Average Technique (CAT). The primary development is based on geo-grid, and it is the recommended option for use.
 