# Multi-Token Prediction

## 一句话理解

Multi-Token Prediction（MTP）是在训练时让模型不仅预测下一个 token，还预测后面多个 token，从而让训练信号更密集，也可能帮助推理加速。

## 最简单的方式

可以增加多个 `lm head`：

- 老版本的语言模型通常只预测下一个 token。
- MTP 可以多加几个可训练的 `lm head`。
- 每个 head 分别预测未来不同位置的 token。
- 训练时，把多个预测位置的交叉熵 loss 组合起来。

## 更复杂的方式

以 DeepSeek-V3 这类设计为例，MTP 不只是多几个线性头，而是增加额外的预测模块。

大致理解：

- 使用当前最后一个 token 的 final hidden state。
- 再结合下一个 token 的 embedding / hidden state 信息。
- 通过额外模块去预测再下一个 token。
- 预测模块里可能包含：
  - concat
  - projector
  - transformer block
  - lm head

## 训练阶段的作用

- 训练信号更密集：一个位置不只监督下一个 token，也监督更远的 token。
- 多个交叉熵 loss 可以线性组合。
- hidden states 可能会被训练得更有“前瞻性”。
- 对推理、规划、代码生成这类需要长程结构的任务可能有帮助。

## 推理阶段的作用

- MTP 可以一次生成多个未来 token 的候选。
- 这些候选通常不能直接无条件采用，需要经过主模型验证。
- 如果验证通过率较高，就可以减少串行 decoding 的步数。
- 这和 speculative decoding 的思想有相似之处：先提出多个候选，再验证。

## 需要注意

- 不能简单理解成“推理时直接拿预测出来的 token 当 embedding 一路滚下去”。
- 更稳妥的说法是：MTP 产生候选 token，主模型再并行验证。
- 实际加速效果取决于：
  - 候选 token 的接受率
  - MTP depth
  - 验证实现
  - batch 和硬件条件

## 我的当前理解

MTP 的价值主要有两点：

- 训练上：让模型学到更密集、更远期的监督信号。
- 推理上：如果候选预测质量高，可以用并行验证减少串行生成步骤。
