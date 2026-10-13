# lmc

## Module Introduction

`lmc` is the lattice Monte Carlo fracture module used to generate crack events and fragment statistics on compact grids. It includes grid generation, fracture simulation kernels, sampler/cache logic, and optional pool-based aggregate reuse for efficient repeated simulations.

## 轴向拉伸方向权重

`LMCSimulator(..., loading_axis="x", loading_kappa=2.0)` 在首步及一般扩展阶段，将平行于加载轴的连接键权重乘以 `exp(loading_kappa)`。轴只支持 `None / "x" / "y"`；`loading_kappa` 是非负有限的无量纲参数。x 向指列方向的 Hbond，y 向指行方向的 Vbond；裂纹段切线与该键方向垂直。实例创建时确定这两个参数，改变配置时重新创建实例。

一般阶段的候选权重为 `STR_e**(-1) * gamma**is_forward * exp(kappa*(bond_normal·loading_axis)**2)`。`gamma` 仍相对于上一裂纹步；`int_bre` 起始直行阶段与自环规则不变。默认 `loading_axis=None, loading_kappa=0` 保持旧算法及随机流，固定种子回归已核对。

这是轴向拉伸的经验取向规则，尚未物理标定，不表示压缩、剪切、加载速度或应力场。方向权重不乘入断裂代价，保留断键代价仍为每条唯一键的 `STR` 之和。`accept_all_cracks=True` 使用现有全域起裂及未分裂损伤保留分支。

首次5%有效母体分裂的试验入口与烟测位于 MUSEN 工作目录的 `scripts/validate_lmc_axial_tension.py`，内部复用 `lmc_mod/effective_split.py`。该观察模式忽略 `NO_FRAG` 停止目标，并核对组件间分割代价与组件内损伤代价。

