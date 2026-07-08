# Faster Attention

## Qwen3-Next / Qwen3.5

- 可能是 `3 * linear attention + full attention` 的混合结构。
- 线性注意力：用 K/V 维护一个记忆矩阵或记忆状态。
- `q` 和记忆矩阵相乘，得到最终结果。
- `gate`：门控，具体作用待核实。

> 待核实：`3 * linear attention + full attention` 的具体说法需要再查官方资料或 config。

## DeepSeek V4

- 也是两种注意力/压缩方式交替。
- 方式 1：较轻压缩 KV，窗口内不压缩，检索窗口外的信息。
- 方式 2：较重压缩 KV，结合 full attention。

> 待核实：这部分是否是 DeepSeek V4 的正式架构描述，后续需要查来源。

## 通用点

- GQA：Grouped-Query Attention，用更少的 K/V head 服务多组 Query，减少 KV cache。
