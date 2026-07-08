# Faster Attention

## 今天的主要收获

最近一些模型会用混合注意力结构来降低长上下文推理成本，同时尽量保留 full attention 的精确记忆能力。

## Qwen3-Next / Qwen3.5 相关速记

- 可能采用类似 `linear attention + full attention` 交替的混合结构。
- 速记理解：若干层线性注意力负责高效维护历史状态，再穿插 full attention 补足精确回忆能力。
- 线性注意力不是完整保存所有 KV cache，而是用 K/V 聚合出一个固定大小的记忆状态。
- 查询时，可以理解为 `q` 和这个记忆状态交互，得到当前 token 对历史信息的读取结果。
- `gate` 可能用于控制当前输入、历史记忆、遗忘/更新之间的比例。

> 待核实：具体层数比例、是否严格是 `3 * linear attention + 1 * full attention`，需要查官方技术报告或模型 config。

## DeepSeek 相关速记

- DeepSeek-V3 中比较确定的注意力相关技术是 MLA（Multi-head Latent Attention）。
- MLA 的核心目标是压缩 KV cache，降低长上下文推理时的显存占用。
- 你今天记到的内容：
  - 一种较轻的 KV 压缩方案：窗口内不压缩，窗口外做检索或压缩。
  - 一种较重的 KV 压缩方案：整体上更依赖压缩后的表示，再结合 full attention。

> 待核实：这部分是否确实属于 DeepSeek-V4 的正式架构描述。目前先保留为“听到的技术路线”，不要直接当作确定事实。

## 通用技术点

### GQA

- GQA 是 Grouped-Query Attention。
- 多个 Query head 共享较少数量的 Key / Value head。
- 主要作用是减少 KV cache 和推理显存，同时尽量保留多头注意力的表达能力。

### Linear Attention

- 标准 full attention 需要显式关注历史 token，复杂度和序列长度强相关。
- linear attention 通过维护固定大小的状态来压缩历史信息。
- 优点是长上下文更省显存、更省计算。
- 缺点是精确回忆能力通常弱于 full attention，所以很多模型会采用 hybrid attention。

### Hybrid Attention

- hybrid attention 指把 linear attention / sparse attention / full attention 等不同注意力形式组合起来。
- 常见思路：
  - 多数层使用更便宜的注意力机制。
  - 少数层保留 full attention，负责精确召回和复杂依赖。
- 这样可以在性能和成本之间做折中。
