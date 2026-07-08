# Multi-Token Prediction

## 核心想法

- Multi-Token Prediction（MTP）让模型一次性预测多个 token。
- 可以理解成在模型上外挂一些预测未来 token 的小组件。

## 最简单的方式

- 老版本的 `lm head` 只预测下一个 token。
- MTP 可以多加几个 `lm head`。
- 这些 `lm head` 也是可训练参数。
- 不同 head 分别往后多预测几个 token。

## 更复杂的方式（DeepSeek-V3）

- 增加一些因果链式关系。
- 用最后一个 token 的 final hidden state，结合下一个 token 直接 embedding 后的 init hidden state，去预测再下一个 token。
- 预测模块可能包括：
  - concat
  - projector
  - transformer block
- 训练时只多往后走一个。

## 好处

- 训练时信号更密集，监督更多。
- 可以把多个交叉熵 loss 线性组合。
- hidden states 会更有前瞻性。
- 推理时可以一次性预测多个 token，再进行验证。
- 验证多个 token 通常比一个一个串行推理更快。

## 注意

- 推理时不是无条件采用预测 token，通常还需要验证。
- 实际加速效果取决于候选 token 的接受率和实现方式。
