# Granulation WMCPBE 与验证脚本使用说明

## 1. WMCPBE 求解器的组成结构

`wmcpbe_granulation` 是一个简化版加权蒙特卡洛 PBE 求解器。

核心入口类位于：

```python
wmcpbe_granulation.mcpbe.MCPBESolver
```

`MCPBESolver` 通过多重继承组合不同功能模块：

```python
class MCPBESolver(MCPBEPost, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
    pass
```

各模块职责如下。

| 文件 | 主要职责 |
| --- | --- |
| `mcpbe.py` | 定义最终入口类 `MCPBESolver`，组合各个 mixin。 |
| `mcpbe_base.py` | 基础框架：构造函数、颗粒初始化入口、采样器初始化、容量扩展、主时间推进循环、`solve_repeats()`、列增删操作。 |
| `mcpbe_initialization.py` | 初始颗粒构造逻辑，包括从显式 `V_flat + W_init` 初始化，以及简单模型初始化。 |
| `mcpbe_agg.py` | 聚并 agglomeration 逻辑：聚并核、聚并倾向率、选择碰撞对、执行一次聚并事件、更新权重和采样器。 |
| `mcpbe_break.py` | 破碎 breakage 逻辑：破碎速率、破碎片段分布 CDF、执行一次破碎事件、更新权重和采样器。 |
| `mcpbe_time_helper.py` | 时间步和权重 packet 的辅助函数，包括聚并/破碎/混合过程的时间步计算。 |
| `reconstruction_mixin.py` | 重构逻辑，用于在计算颗粒数过多时压缩/重采样颗粒表示。 |
| `mcpbe_post.py` | 后处理：按时间快照计算矩、PSD CDF 等。 |
| `fenwick.py` | Fenwick tree 采样器，用于根据事件倾向率快速随机抽样。 |

## 2. 基本运行顺序：以 `solve_repeats()` 为例

### 2.1 为什么 `solve_repeats()` 

`solve_repeats()` 是目前使用 WMCPBE 最简单、最稳定的外部接口。它会自动完成以下工作：

1. 为每次重复计算生成随机种子。
2. 深拷贝当前 solver 模板。
3. 为每个拷贝设置独立随机数生成器。
4. 用同一组初始颗粒和权重初始化每个 solver。
5. 调用内部 `solve()` 完成一次蒙特卡洛轨迹。
6. 调用后处理方法计算随时间变化的矩。
7. 汇总所有重复运行的结果。

蒙特卡洛方法本质上是随机方法。单次运行是一条随机轨迹，可能带有明显随机波动。重复计算的目的包括：

- 降低随机误差；
- 估计结果方差或标准差；
- 增加统计可信度；
- 比较不同参数组时避免把单条轨迹误认为模型趋势。

因此，一般建议用 `solve_repeats(N=...)`，而不是只调用一次 `solve()`。

### 2.2 推荐的最小调用方式

外部使用时，通常先创建一个不自动初始化的 solver 模板：

```python
from wmcpbe_granulation import MCPBESolver

solver = MCPBESolver(
    dim=2,
    t_vec=t_vec,
    verbose=False,
    load_attr=False,
    init=False,
)
```

然后显式设置参数，例如：

```python
solver.process_type = "breakage"      # "agglomeration" | "breakage" | "mix"
solver.COLEVAL = 3                    # 聚并核类型
solver.CORR_BETA = 1e-3               # 聚并核系数
solver.BREAKRVAL = 1                  # 破碎速率模型
solver.BREAKFVAL = 2                  # 破碎片段分布模型
solver.pl_P1 = 3e-2
solver.pl_P2 = 1.0
solver.pl_P3 = 3e-2
solver.pl_P4 = 1.0
solver.pl_v = 1.0
solver.pl_q = 1.0
solver.G = 1.0
solver.alpha_prim = np.ones(dim ** 2)
```

最后用显式初始颗粒和权重运行(需要显示定义V_flat和W_init，后边会讲)：

```python
results, psd_info = solver.solve_repeats(
    N=10,
    base_seed=42,
    maxiter=int(1e9),
    init_Vc=False,
    Vc=Vc,
    V_flat=V_flat,
    W_init=W_init,
)
```

### 2.3 `solve_repeats()` 主要参数

| 参数 | 含义 |
| --- | --- |
| `N` | 重复计算次数。`N` 越大，统计平均越稳定，但计算成本越高。 |
| `base_seed` | 基础随机种子。若不传 `seeds`，会用它生成 `N` 个独立种子。 |
| `seeds` | 可选的显式种子序列。长度必须等于 `N`。 |
| `maxiter` | 每次重复计算允许的最大事件数，用于防止无限长计算。 |
| `init_Vc` | 是否由模型内部根据 `a0/n0` 初始化控制体积。使用显式 `Vc` 时设为 `False`。 |
| `Vc` | 控制体积。使用 `V_flat + W_init` 外部初始化时通常需要传入。 |
| `V_flat` | 初始颗粒体积数组，形状为 `(dim + 1, N_particles)`。前 `dim` 行为各组分体积，最后一行为总颗粒体积。 |
| `W_init` | 每个计算颗粒的权重，表示该计算颗粒代表多少个真实颗粒。长度必须等于 `V_flat` 的列数。 |
| `psd_enable` | 是否计算 PSD 后处理。默认 `False`。 |
| `psd_basis` | PSD 的权重基准，通常为 `"volume"` 或 `"number"`。 |
| `psd_x_grid` | 若给定，则输出固定粒径网格上的 `Q(x)`。 |
| `psd_Q_grid` | 若给定，则输出固定分位数上的 `x(Q)`。 |

返回值：

```python
results, psd_info
```

其中 `results` 是列表，每个元素对应一次重复运行：

```python
{
    "seed_info": ...,
    "t_vec": ...,
    "moments": ...,
}
```

`moments` 的形状通常为 `(3, 3, T)`，即 `mu[i, j, t]`。1D 情况主要使用 `mu[i, 0, t]`。

## 3. 聚并、破碎、时间推进、权重、batch-wise 与重构

### 3.1 权重的基本含义

WMCPBE 中的一个“计算颗粒”不一定只代表一个真实颗粒。它可以带有权重 `W[k]`：

```python
W[k] = 第 k 个计算颗粒代表的真实颗粒数量
```

这使得 solver 可以用较少的计算颗粒代表大量真实颗粒。进行采样，计算矩时会使用权重，例如 1D 下矩为：

```python
mu_i = sum_k W[k] * V[k]**i / Vc
```

2D 下：

```python
mu_ij = sum_k W[k] * V1[k]**i * V3[k]**j / Vc
```

### 3.2 batch-wise / packet 事件

普通蒙特卡洛事件可以理解为“一次事件处理一个真实颗粒或一对真实颗粒”。加权版本为了提高效率，允许一次事件代表一小批真实事件，即 packet 或 batch-wise 事件。

两个关键参数是：

```python
agg_dW_max
break_dW_max
```

它们控制每次聚并/破碎事件最多消耗多少权重。

- `agg_dW_max = 1.0`：一次聚并事件最多代表 1 个真实聚并 packet。
- `break_dW_max = 1.0`：一次破碎事件最多代表 1 个真实破碎 packet。
- 更大的值会让每个事件推进更多真实权重，通常更快，但统计/数值近似也更粗。

在 `simple_validation.py` 中，如果希望接近普通逐事件计算，可以把二者都设为：

```python
"agg_dW_max": 1.0,
"break_dW_max": 1.0,
```

这就是文档和示例里说的 batch-wise 关闭或最小化的常用方式。

### 3.3 聚并 agglomeration 的大概流程

聚并逻辑主要在 `mcpbe_agg.py` 中。

一次聚并事件大致如下：

1. 根据当前颗粒状态计算每个颗粒的聚并倾向率 `_r_agg`。
2. 用 Fenwick sampler 根据 `_r_agg` 随机选择第一个颗粒 `i`。
3. 根据聚并核、权重和接受概率选择第二个颗粒 `j`。
4. 根据 `agg_dW_max / agg_dW_min / agg_dW_mode` 计算本次事件的权重包大小 `dW`。
5. 创建一个新颗粒，其组分体积为 `Vi + Vj`，权重为 `dW`。
6. 从父颗粒权重中扣除 `dW`；如果是自聚并，则同一个父颗粒扣除 `2*dW`。
7. 如果父颗粒权重降到 0 或以下，则删除该计算颗粒。
8. 重建聚并倾向率和采样器。
9. 若当前是 `mix` 过程，还会更新破碎采样器。

### 3.4 破碎 breakage 的大概流程

破碎逻辑主要在 `mcpbe_break.py` 中。

一次破碎事件大致如下：

1. 计算每个颗粒的破碎速率和破碎倾向率 `_break_rate`。
2. 根据 `_break_rate` 随机选择一个母颗粒 `k`。
3. 根据 `break_dW_max` 和当前母颗粒权重确定本次破碎 packet 大小 `dW`。
4. 根据 `BREAKFVAL / pl_v / pl_q` 构造或复用破碎片段分布 CDF。
5. 从母颗粒体积中随机生成多个碎片。
6. 每个碎片作为新计算颗粒加入，权重为 `dW`。
7. 母颗粒权重减少 `dW`；若剩余权重为 0，则删除母颗粒。
8. 更新破碎采样器；若当前是 `mix` 过程，还会更新聚并采样器。

### 3.5 时间推进逻辑

时间推进逻辑主要在 `mcpbe_base.py` 和 `mcpbe_time_helper.py`。

核心思想：

- 当前所有可能事件的倾向率之和决定下一次事件的时间尺度。
- 聚并、破碎、混合过程分别有对应的时间步策略。
- `solve()` 内部根据 `process_type` 选择执行聚并、破碎或二者竞争。
- 每次事件发生后，更新当前时间 `current_time`，并在跨过保存时间点时保存快照。

`process_type` 支持：

```python
"agglomeration"
"breakage"
"mix"
```

### 3.6 reconstruction 是做什么的

重构逻辑在 `reconstruction_mixin.py` 中。

随着模拟推进，计算颗粒数可能快速增长。例如破碎会不断生成新碎片，聚并/混合过程也可能导致状态分布变复杂。重构的目标是：

- 控制计算颗粒数量；
- 将大量相近颗粒压缩成较少代表颗粒；
- 尽量保持关键矩或分布特征；
- 降低后续计算成本。

常用参数：

| 参数 | 含义 |
| --- | --- |
| `recon_enable` | 是否启用重构。 |
| `recon_method` | 重构方法，如 `"RS"`、`"2PM"`、`"4PM"`、`"4PMC"`、`"QMX"`。 |
| `recon_N_max` | 当活跃计算颗粒数超过该值时触发重构。 |
| `recon_bins` | 重构网格数。 |
| `recon_RS_target` | RS 重采样目标颗粒数。 |

启用重构示例：

```python
"recon_enable": True,
"recon_method": "4PMC",
"recon_N_max": 4000,
"recon_bins": 30,
"recon_RS_target": 1000,
```

关闭重构示例：

```python
"recon_enable": False,
```

注意：通常情况下使用`4PMC`即可，该方法综合精度最高。

## 4. 添加自己的 process：以 nucleation 为例

如果要添加新过程，例如 `nucleation`，建议沿着现有聚并/破碎结构扩展，而不是把逻辑直接塞进 `solve()`。

推荐实现路径如下。

### 4.1 新增过程模块

建议新增类似文件：

```text
mcpbe_nucleation.py
```

并定义 mixin：

```python
class MCPBENucleation:
    ...
```

该模块建议包含：

- nucleation 参数准备方法；
- nucleation 倾向率或总速率计算方法；
- 单次 nucleation 事件执行方法，例如 `_do_one_nucleation()`；
- nucleation 相关采样器或速率数组维护方法。

### 4.2 修改入口类继承

在 `mcpbe.py` 中把新 mixin 加入 `MCPBESolver`：

```python
class MCPBESolver(MCPBEPost, MCPBENucleation, MCPBEBreak, MCPBEAgg, MCPBEBase, ReconstructionMixin):
    pass
```

继承顺序应当保证 `MCPBEBase` 能调用到新 mixin 中的方法。

### 4.3 修改 `process_type` 支持

需要在以下位置扩展 `process_type` 判断：

- `mcpbe_base.py`
  - `_initialize_samplers()`
  - `solve()`
  - `_maybe_double_control_volume()` 如适用
  - debug/状态检查逻辑如适用
- `mcpbe_time_helper.py`
  - 添加 nucleation 的时间步策略，例如 `_build_nucleation_dt_strategy()`。

如果要支持组合过程，可能需要定义新的模式：

```python
"nucleation"
"nucleation_agglomeration"
"nucleation_breakage"
"mix"
```

### 4.4 权重更新

新 process 的核心难点通常不是“生成新颗粒”，而是保持权重和矩的一致。

添加新事件时必须明确：

- 新颗粒体积是多少；
- 新颗粒权重是多少；
- 是否消耗旧颗粒权重；
- 是否改变控制体积 `Vc`；
- 事件发生后哪些速率数组需要更新；
- 哪些采样器需要 rebuild 或局部 update；
- 对 `V_save / W_save / Vc_save` 后处理快照是否有影响。

例如 nucleation 如果表示“从连续相中新生成颗粒”，可能不消耗已有颗粒权重，但会增加新的计算颗粒：

```python
self._append_particle_column(new_volume)
self.W[new_idx] = dW_nuc
```

随后需要更新：

- 聚并速率 `_r_agg`，因为新颗粒会参与聚并；
- 破碎速率 `_break_rate`，如果新颗粒也可破碎；
- 对应 Fenwick sampler；
- 必要时触发 reconstruction。

## 5. `granulation/validation.py` 的基本思路

`granulation/validation.py` 是一个只面向 `wmcpbe_granulation` 的轻量验证包装脚本。

它主要用于：

- 用同一初始颗粒状态运行多个 WMCPBE 参数组；
- 比较不同 batch-wise 参数、重构参数、随机种子、重复次数的影响；
- 与已有解析解比较矩的演化趋势；
- 生成后处理图像。

主要类如下。

| 类 | 作用 |
| --- | --- |
| `CaseConfig` | 定义物理问题：维度、核函数、过程类型、时间网格、解析解参数、初始颗粒规模。 |
| `WMCPBEVariantConfig` | 定义一个 WMCPBE 参数组，包括重复次数、随机种子、最大事件数和 solver 属性覆盖。 |
| `GranulationValidationConfig` | 汇总 case 和多个 WMCPBE variants。 |
| `GranulationValidationRunner` | 运行所有 WMCPBE variants，并生成解析解。 |
| `ValidationResult` | 保存时间、初始状态、所有方法结果。 |
| `ValidationPlotter` | 绘制矩和总体积图。 |

## 6. `simple_validation.py` 的使用方式

`simple_validation.py` 是一个最小示例，用来展示如何配置 case 和多个 WMCPBE 参数组。

基本结构：

```python
case = CaseConfig(...)

wmcpbe_variants = [
    WMCPBEVariantConfig(...),
    WMCPBEVariantConfig(...),
]

config = GranulationValidationConfig(
    case=case,
    wmcpbe_variants=wmcpbe_variants,
    verbose=False,
)

result = GranulationValidationRunner(config).run()
plotter = ValidationPlotter(result)
plotter.plot_all_moments(relative=True, include_total_volume=False)
plotter.show()
```

### 6.1 `CaseConfig` 重要参数

| 参数 | 含义 |
| --- | --- |
| `dim` | 维度，支持 `1` 或 `2`。 |
| `kernel` | 聚并核类型，目前包装脚本支持 `"const"` 和 `"sum"`。 |
| `process` | 过程类型：`"agglomeration"`、`"breakage"`、`"mix"`。 |
| `t_vec` | 输出/保存时间网格。 |
| `x` | 用于构造示例初始颗粒体积的特征粒径。 |
| `beta0` | 解析解和聚并核参数。 |
| `g` | 剪切率或相关核参数。 |
| `p1`、`p2` | 破碎速率参数。2D 中 `pl_P3/pl_P4` 默认跟 `p1/p2` 一致。 |
| `pl_v`、`pl_q` | 破碎片段分布参数。 |
| `initial_number_density` | 初始数密度，脚本用它计算 `Vc`。 |
| `initial_total_weight` | 初始计算颗粒代表的真实颗粒总权重。 |

### 6.2 `WMCPBEVariantConfig` 重要参数

| 参数 | 含义 |
| --- | --- |
| `name` | 图例和结果字典中的名称。 |
| `repeats` | 重复计算次数。统计比较时建议大于 1。 |
| `base_seed` | 基础随机种子。 |
| `maxiter` | 单次重复最大事件数。 |
| `enabled` | 是否启用该参数组。 |
| `attrs` | 覆盖到 `MCPBESolver` 上的参数字典。 |

`attrs` 是最常改的位置。例如关闭 reconstruction：

```python
attrs={
    "recon_enable": False,
    "break_dW_max": 1.0,
    "agg_dW_max": 1.0,
}
```

启用 reconstruction：

```python
attrs={
    "recon_enable": True,
    "recon_method": "4PMC",
    "recon_N_max": 4000,
    "recon_bins": 30,
    "recon_RS_target": 1000,
    "break_dW_max": 1.0,
    "agg_dW_max": 1.0,
}
```

改变 batch-wise 强度：

```python
attrs={
    "break_dW_max": 10.0,
    "agg_dW_max": 20.0,
}
```

如果想尽量接近逐个真实事件处理，则使用：

```python
attrs={
    "break_dW_max": 1.0,
    "agg_dW_max": 1.0,
}
```

## 7. validation 中的初始颗粒输入方式

granulation 版本目前采用显式颗粒输入：

```python
V_flat + W_init + Vc
```

这也是外部开发最容易理解和修改的方式。

### 7.1 输入位置

默认初始颗粒定义在：

```python
GranulationValidationRunner.build_example_initial_particles()
```

该方法返回：

```python
InitialParticleState(
    Vc=...,
    V_flat=...,
    W_init=...,
)
```

随后在 `run_wmcpbe_variant()` 中传给：

```python
solver.solve_repeats(
    init_Vc=False,
    Vc=initial_state.Vc,
    V_flat=initial_state.V_flat,
    W_init=initial_state.W_init,
)
```

### 7.2 `V_flat` 的格式

`V_flat` 必须是二维数组：

```python
V_flat.shape == (dim + 1, N_particles)
```

1D 情况：

```python
V_flat[0, k] = 第 k 个颗粒的体积
V_flat[1, k] = 第 k 个颗粒的总量体积
```

由于 1D 只有一个组分，则：

```python
V_flat[1, :] = V_flat[0, :]
```

2D 情况：

```python
V_flat[0, k] = 第 k 个颗粒中组分 1 的体积
V_flat[1, k] = 第 k 个颗粒中组分 2 的体积
V_flat[2, k] = 第 k 个颗粒总量体积 = V_flat[0, k] + V_flat[1, k]
```

### 7.3 `W_init` 的格式

`W_init` 是一维数组：

```python
W_init.shape == (N_particles,)
```

含义是：

```python
W_init[k] = 第 k 个计算颗粒代表的真实颗粒数
```

例如 4 个计算颗粒代表不同数量的真实颗粒：

```python
W_init = np.array([40000.0, 20000.0, 20000.0, 20000.0])
```

### 7.4 `Vc` 的含义

`Vc` 是控制体积。矩后处理时会用它归一化：

```python
mu00(0) = sum(W_init) / Vc
```

当前示例中：

```python
Vc = initial_total_weight / initial_number_density
```

因此 `initial_number_density` 控制初始 `mu00`。

### 7.5 当前示例规则

1D 示例：

```python
component_volumes = np.array([[v0, 2*v0, 4*v0]])
fractions = np.array([0.60, 0.30, 0.10])
```

2D 示例：

```python
component_volumes = np.array([
    [v0, 2*v0, v0, 2*v0],
    [v0, v0, 2*v0, 2*v0],
])
fractions = np.array([0.40, 0.20, 0.20, 0.20])
```

脚本会把 `fractions` 转换为权重：

```python
W_init = initial_total_weight * fractions
```

### 7.6 如何定义自己的输入，例如已知 PSD

如果你已有一个 PSD 或外部颗粒分布，推荐改写：

```python
build_example_initial_particles()
```

一般步骤：

1. 把 PSD 离散成若干代表颗粒。
2. 为每个代表颗粒定义组分体积。
3. 根据 PSD 的 number fraction 或 volume fraction 计算 `W_init`。
4. 组装 `V_flat`。
5. 根据想要的初始数密度定义 `Vc`。
6. 返回 `InitialParticleState`。

伪代码：

```python
def build_example_initial_particles(self) -> InitialParticleState:
    case = self.config.case

    # 1. 用户自己的 PSD 离散点
    particle_volumes = ...
    particle_weights = ...

    # 2. 组装 V_flat
    V_flat = np.zeros((case.dim + 1, particle_volumes.shape[1]))
    V_flat[:case.dim, :] = particle_volumes
    V_flat[-1, :] = np.sum(particle_volumes, axis=0)

    # 3. 定义控制体积
    Vc = np.sum(particle_weights) / desired_number_density

    return InitialParticleState(
        Vc=Vc,
        V_flat=V_flat,
        W_init=particle_weights,
    )
```
