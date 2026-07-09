# Paper Case Scripts

The numeric prefix at the beginning of each script filename corresponds to the section number in the paper.

All scripts in this folder can be run directly and correspond one-to-one to the cases reported in the paper. For example, scripts starting with `31`, `32`, or `36` are associated with Sections 3.1, 3.2, and 3.6, respectively.

Except for the ten random seeds discussed in Section 3.6, changing the random seed is straightforward. Edit the `base_seed` value in `WMCPBEVariantConfig` before running the script:

```python
WMCPBEVariantConfig(
    base_seed=12345,
)
```
