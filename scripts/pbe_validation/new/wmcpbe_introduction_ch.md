# WMCPBE 验证脚本使用说明

本文档对应介绍本目录下三个与 WMCPBE 相关的新脚本：

- `advance_validation`
- `reconstruction_monitor`
- `WMCPBE_sensitivity_analysis`

文档按照统一顺序组织：

1. 脚本的用途
2. 脚本需要哪些输入
3. 这些输入应如何配置
4. 脚本最终会输出什么


## 1. 共用设计逻辑

这几个脚本都遵循相近的总体流程：

1. 定义物理问题
2. 定义一个参考 dPBE 配置
3. 定义一个或多个 WMCPBE 配置
4. 构建统一的初始条件
5. 运行对应的求解流程
6. 与某个参考量进行比较
7. 将所有关键数据保存为 Excel，便于后处理

这些脚本主要依赖以下公共模块：

- `validation.py`
  - 提供公共配置对象
  - 提供标准的 `ValidationRunner`
- `pbe_validation_advance.py`
  - 提供二维 Dirichlet 型初始条件的高级流程
- `plotter_new.py`
  - 提供统一的论文风格绘图接口

几个脚本中最常用的配置对象包括：

- `CaseConfig`
- `DPBEVariantConfig`
- `WMCPBEVariantConfig`
- 在二维高级流程中还会用到 `DirichletInitialCondition`


## 2. 基础输入对象

### 2.1 `CaseConfig`

`CaseConfig` 用于定义“我们到底在求解什么物理问题”。

常用字段包括：

- `dim`
  - 颗粒空间维度
  - `1` 表示一维
  - `2` 表示二维
- `kernel`
  - 例如 `"const"` 或 `"sum"`
- `process`
  - `"agglomeration"`
  - `"breakage"`
  - `"mix"`
- `t_vec`
  - 时间向量
- `x`
  - 初始颗粒尺度或参考尺度
- `beta0`
  - 聚并核参数
- `p1`, `p2`
  - 破碎相关参数
- `use_psd`
  - dPBE 是否使用 PSD 文件作为初始化

可以把它理解为：

“物理问题说明书”


### 2.2 `DPBEVariantConfig`

该对象定义参考 dPBE 的离散方式。

常用字段：

- `name`
- `grid`
- `ns`
- `s`

在这些新脚本中，dPBE 主要起两种作用：

- 用来构建统一的 canonical 初始条件
- 在高级验证中作为数值参考之一


### 2.3 `WMCPBEVariantConfig`

该对象定义一个 WMCPBE 求解设置。

常用字段：

- `name`
- `repeats`
- `base_seed`
- `maxiter`
- `enabled`
- `attrs`

其中最重要的是 `attrs`，因为大多数 WMCPBE 参数都通过它传入。

典型的 `attrs` 内容包括：

- `a0`
- `V_eff_init`
- `recon_enable`
- `recon_method`
- `recon_bins`
- `recon_N_max`
- `recon_RS_target`
- `break_dW_min`
- `break_dW_max`
- `agg_dW_min`
- `agg_dW_max`

可以把它理解为：

“这一套 WMCPBE 数值实验怎么跑”


### 2.4 `DirichletInitialCondition`

这个对象用于二维高级流程中的初始条件定义。

字段包括：

- `alpha_x`
- `alpha_y`
- `alpha_rest`
- `x_min_scale`
- `x_max_scale`
- `y_min_scale`
- `y_max_scale`
- `total_number`
- `volume_concentration`

它描述的是定义在 dPBE 网格范围上的一个截断二维 Dirichlet 型分布。

其中有一个重要规则：

- 如果同时给了 `total_number` 和 `volume_concentration`
  - 则优先使用 `volume_concentration`


## 3. `advance_validation`

### 3.1 用途

`advance_validation.py` 是主要的高级验证流程，用于比较：

- 解析解
- dPBE
- 一个或多个 WMCPBE 结果

它主要针对二维问题，并且支持更完整的后处理。


### 3.2 主要组成

这个流程主要由以下对象驱动：

- `PBEValidationAdvanced`
- `Dirichlet2DValidationRunner`

最直接的入口脚本是：

- `advance_validation.py`


### 3.3 输入顺序

使用 `advance_validation` 时，建议按照下面顺序组织输入。

#### 第 1 步：定义物理问题

先构造 `CaseConfig`。

典型示例如下：

```python
case = CaseConfig(
    dim=2,
    kernel="const",
    process="breakage",
    t_vec=np.arange(0.0, 10.0 + 1e-12, 1.0),
    x=2e-1,
    beta0=1e-3,
    p1=3e-2,
    p2=1.0,
    use_psd=False,
)
```


#### 第 2 步：定义一个参考 dPBE 配置

在 advanced 流程中，通常只启用一个 dPBE 配置。

典型示例如下：

```python
dpbe_variants = [
    DPBEVariantConfig(name="dPBE", grid="geo", ns=15, s=2),
]
```


#### 第 3 步：定义一个或多个 WMCPBE 配置

典型示例如下：

```python
wmcpbe_variants = [
    WMCPBEVariantConfig(
        name="WMCPBE (fine)",
        repeats=4,
        attrs={
            "a0": 100000,
            "V_eff_init": 1000,
            "recon_N_max": 4000,
            "recon_bins": 30,
            "recon_method": "4PMC",
        },
    ),
    WMCPBEVariantConfig(
        name="WMCPBE (ref)",
        repeats=5,
        attrs={
            "a0": 100000,
            "V_eff_init": 1000,
            "recon_N_max": 4000,
            "recon_bins": 30,
            "recon_method": "4PMC",
            "break_dW_max": 10,
            "agg_dW_max": 5,
        },
    ),
]
```


#### 第 4 步：构造完整的 `ValidationConfig`

典型示例如下：

```python
config = ValidationConfig(
    case=case,
    dpbe_variants=dpbe_variants,
    wmcpbe_variants=wmcpbe_variants,
    qmom_variants=[],
    reference_dpbe_name="dPBE",
)
```


#### 第 5 步：定义初始分布

典型示例如下：

```python
init_dist = DirichletInitialCondition(
    alpha_x=2.5,
    alpha_y=3.0,
    alpha_rest=4.0,
    x_min_scale=2.0,
    x_max_scale=0.5,
    y_min_scale=2.0,
    y_max_scale=0.5,
    total_number=5e5,
    volume_concentration=None,
)
```


#### 第 6 步：创建对象并运行

```python
advanced = PBEValidationAdvanced(config=config, init_dist=init_dist)
result = advanced.run()
```


### 3.4 可用输出

运行完成后，常见调用流程如下：

```python
advanced.print_moment_error_summary(result)
advanced.plot_selected_moments(result, relative=True)
advanced.plot_psd_snapshot(result, t_index=-1, two_d=False, marginal=True, total=True)
advanced.plot_error_time_pareto(result)
advanced.plot_moment_variances(result)
advanced.show()
```

各个方法的含义：

- `print_moment_error_summary`
  - 打印各选定矩的误差摘要
- `plot_selected_moments`
  - 绘制 `M00`、`M01`、`M11`、`M02` 随时间变化
- `plot_psd_snapshot`
  - 输出 PSD 快照
  - 支持三种模式：
    - `two_d`
    - `marginal`
    - `total`
- `plot_error_time_pareto`
  - 绘制误差-CPU 时间 Pareto 图
- `plot_moment_variances`
  - 绘制 WMCPBE 的归一化方差


### 3.5 Excel 输出

在 advanced 流程中，每个 `print/plot` 方法都会在真正输出前先写一个 Excel 文件。

默认导出位置：

- `scripts/pbe_validation/new/exports`

Excel 中通常包括：

- `metadata` 表
- 各类曲线/曲面/标签/误差数据表

这些数据专门为了后续在 Origin、Excel 或其他后处理工具中继续加工。


## 4. `reconstruction_monitor`

### 4.1 用途

`reconstruction_monitor.py` 用于分析特殊调试版本求解器中，重构操作本身带来的误差：

- `wmcpbe_recon_debug`

它关心的不是整个求解器随时间的总误差，而是：

“每一次 reconstruction 这一动作本身引入了多少误差”

它监测的量包括：

- `M00` 相对误差
- `M01` 相对误差
- `M11` 相对误差
- `M02` 相对误差
- 重构后 `N(x, y)` 相对重构前 `N_ref(x, y)` 的 L1 误差


### 4.2 内部机制

这个流程依赖于以下文件中新增的监测逻辑：

- `mcpbe/src/wmcpbe_recon_debug/reconstruction_mixin.py`

每当求解器真正执行一次 reconstruction 时，都会记录：

- 当前重构发生的时间
- 对应事件号
- 重构前后粒子数
- 重构前矩
- 重构后矩
- 各矩相对误差
- 统一网格下 `N(x, y)` 的 L1 误差


### 4.3 输入顺序

它的输入结构刻意和 `advance_validation` 接近。

#### 第 1 步：定义 `CaseConfig`

这里要求是二维问题。

#### 第 2 步：定义一个启用的 dPBE 参考配置

这个 dPBE 仍然用于构建 canonical 初值。

#### 第 3 步：定义一个或多个 WMCPBE 配置

这里建议重点设置 reconstruction 相关参数，例如：

- `recon_method`
- `recon_bins`
- `recon_N_max`
- `recon_monitor_psd_bins`

#### 第 4 步：定义初始分布

与 advanced 一样，使用 `DirichletInitialCondition`。

#### 第 5 步：创建监测器并运行

```python
monitor = ReconstructionMonitor(config=config, init_dist=init_dist)
result = monitor.run()
```


### 4.4 可用输出

典型调用方式如下：

```python
monitor.print_summary(result)
monitor.plot_moment_errors(result)
monitor.plot_psd_l1_error(result)
monitor.show()
```

这些方法的含义：

- `print_summary`
  - 打印各个误差量的最大值和最终值
- `plot_moment_errors`
  - 绘制 reconstruction 对全局矩带来的误差随时间变化
- `plot_psd_l1_error`
  - 绘制 reconstruction 对 `N(x, y)` 带来的 L1 误差随时间变化


### 4.5 Excel 输出

默认导出位置：

- `scripts/pbe_validation/new/exports_reconstruction_monitor`

其中有一个特别重要的点：

summary 输出的 Excel 除了平均统计外，还包含一个 `raw_events` 表，里面保存了：

- 每个 run
- 每次 reconstruction
- 对应的原始误差数据

这个表很适合：

- 自己重新计算统计量
- 比较早期和晚期重构误差
- 在 Origin 中做事件级分析


## 5. `WMCPBE_sensitivity_analysis`

### 5.1 用途

`WMCPBE_sensitivity_analysis.py` 用于对 WMCPBE 的关键参数做方差型敏感度分析。

当前默认分析的误差指标是：

- 综合矩误差 `aggregated_moment_error`

脚本的结构已经预留好了接口，后面可以继续增加其他误差指标。


### 5.2 主体思路

该流程基于以下要素：

- 一个物理问题 `CaseConfig`
- 一个参考 dPBE 配置
- 一个启用的 WMCPBE 模板配置
- 一组待分析参数
- `SALib` 的 Sobol/Saltelli 采样与分析

对于每一个采样点，流程会：

1. 从模板 WMCPBE 配置生成一套新的参数
2. 调用 `ValidationRunner`
3. 计算误差指标
4. 保存该采样点的误差值

最后再根据所有样本结果计算 Sobol 指数。


### 5.3 输入顺序

#### 第 1 步：定义 `CaseConfig`

它定义了要分析的物理问题。

#### 第 2 步：定义一个参考 dPBE 配置

因为 `ValidationRunner` 仍然依赖 dPBE 来构建 canonical 初始条件，所以这一项必须保留。

#### 第 3 步：定义一个且仅一个启用的 WMCPBE 模板配置

这是非常重要的约束。

敏感度分析脚本要求：

- `config.wmcpbe_variants` 中只能有一个启用的 WMCPBE 配置

因为它会把这个配置当作模板，然后不断修改其中的参数。

典型示例如下：

```python
wmcpbe_variants = [
    WMCPBEVariantConfig(
        name="WMCPBE template",
        repeats=4,
        attrs={
            "a0": 60000,
            "V_eff_init": 1000,
            "recon_enable": True,
            "recon_method": "4PMC",
            "break_dW_min": 10.0,
            "break_dW_max": 10.0,
            "agg_dW_min": 10.0,
            "agg_dW_max": 10.0,
            "recon_bins": 24,
            "recon_N_max": 3500,
            "recon_RS_target": 1200,
        },
    ),
]
```


#### 第 4 步：定义敏感度参数

每个参数用 `SensitivityParameter` 来描述。

字段包括：

- `name`
- `bounds`
- `kind`
- `targets`

例如：

```python
parameters = [
    SensitivityParameter(
        name="delta_w",
        bounds=(2.0, 30.0),
        kind="float",
        targets=("break_dW_min", "break_dW_max", "agg_dW_min", "agg_dW_max"),
    ),
    SensitivityParameter(
        name="recon_bins",
        bounds=(10.0, 40.0),
        kind="int",
        targets=("recon_bins",),
    ),
    SensitivityParameter(
        name="recon_N_max",
        bounds=(1500.0, 6000.0),
        kind="int",
        targets=("recon_N_max",),
    ),
]
```

这里的含义是：

- `delta_w` 这一项实际上会同时作用到多个求解器属性
- `recon_bins` 和 `recon_N_max` 各自只对应一个求解器属性


#### 第 5 步：创建分析器

典型示例如下：

```python
analyzer = WMCPBESensitivityAnalyzer(
    config=config,
    parameters=parameters,
    metric_name="aggregated_moment_error",
    sample_size=16,
    calc_second_order=False,
)
```

几个关键字段说明：

- `metric_name`
  - 当前推荐并默认使用 `"aggregated_moment_error"`
- `sample_size`
  - Sobol/Saltelli 的基础样本数
- `calc_second_order`
  - 是否进一步计算二阶 Sobol 指数


#### 第 6 步：运行

```python
result = analyzer.run()
```


### 5.4 当前误差接口

目前内置的误差接口是：

- `aggregated_moment_error`

它的参考对象是解析解。

当前实现中：

- 一维时使用 `M00`、`M10`、`M20`
- 二维时使用 `M00`、`M01`、`M11`、`M02`

这个脚本的设计是开放的，后面可以继续加，例如：

- 最大相对矩误差
- 最终时刻矩误差
- PSD 的 L1 误差
- reconstruction 层面的误差


### 5.5 输出

当前脚本会在控制台打印：

- 一阶 Sobol 指数
- 总效应 Sobol 指数

并导出一个 Excel 文件，其中包括：

- `metadata`
- `sample_records`
- `first_order`
- `total_order`
- 如果启用，则还有 `second_order`

默认导出位置：

- `scripts/pbe_validation/new/exports_wmcpbe_sensitivity`


## 6. 推荐使用顺序

如果你不确定应该先用哪个脚本，通常建议按下面顺序进行。

### 第一步：先用 `advance_validation`

适合回答的问题：

- 这一套 WMCPBE 是否足够准确？
- 它与 dPBE、解析解相比如何？
- 矩、PSD、方差随时间如何演化？


### 第二步：再用 `reconstruction_monitor`

适合回答的问题：

- reconstruction 本身引入了多少误差？
- 哪些 reconstruction 参数会导致更大局部误差？
- 总误差中有多少来自 reconstruction？


### 第三步：最后用 `WMCPBE_sensitivity_analysis`

适合回答的问题：

- 哪个 WMCPBE 参数最关键？
- 解对 `ΔW`、`recon_bins`、`recon_N_max`、`a0` 谁更敏感？
- 后续应该优先调哪个参数？


## 7. 实际使用建议

### 7.1 关于重复次数和开销

对于基于 WMCPBE 的流程：

- `repeats` 越多，统计越稳
- 但敏感度分析会很快变得昂贵

比较实用的方式是：

1. 先用较小 `repeats` 和较小样本数调通流程
2. 确认脚本和参数设置都正确后，再逐步增加


### 7.2 关于“先导出再后处理”

这些新脚本都采用了 export-first 的思路：

- 每个 plot/print 方法都会先写 Excel
- 图像只是基于这些数值的一个展示层

这对后续使用以下工具非常友好：

- Origin
- Excel
- MATLAB
- 自己写的统计脚本


### 7.3 关于参考对象

不同脚本里的参考对象并不完全相同，使用时要注意：

- 在 advanced 的矩误差比较中：
  - 参考是解析解
- 在 PSD snapshot 比较中：
  - 可以使用 `WMCPBE (ref)` 作为 PSD 参考
- 在 reconstruction monitor 中：
  - 参考是每次 reconstruction 前的状态
- 在 sensitivity analysis 中：
  - 标量误差指标目前仍然是相对解析解定义的


## 8. 总结

这三个脚本分别回答三个层次的问题：

- `advance_validation`
  - 看整个求解器的精度与比较结果
- `reconstruction_monitor`
  - 看 reconstruction 这一步单独带来的误差
- `WMCPBE_sensitivity_analysis`
  - 看 WMCPBE 参数的重要性排序

如果把它们串起来使用，一个比较自然的研究流程是：

1. 先验证整体精度
2. 再拆分重构误差来源
3. 最后分析参数敏感度

这也是目前最推荐的 WMCPBE 系统化研究路径。
