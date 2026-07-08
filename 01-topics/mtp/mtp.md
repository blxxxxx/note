multi-token prediction

根据一些外挂的小组件使得模型一次性的可以预测多个token

最简单的方式 多一些lm head（linear）
- 老版本的lm head预测一个
- 多搞几个lm head也是可训练参数，往后多预测

比较复杂的方式 增加一些因果链式关系 （dsv3）
- 用最后一个token的final hidden states和下一个token直接embedding后的init hidden states去预测再下一个token的输出
- 预测模块有concat projector transformerblock等较为复杂的模块
- 推理时拿预测出来的作为embedding
- 实际训练时只多往后走一个

好处
- 训练时是的训练信号更密集，监督更多，多个交叉熵线性组合；使得hidden states更具有前瞻性
- 推理时好处是，一次性预测更多个后进行验证时验证成本比一个一个推理要快，flops几乎一致，但是把多次向量运算变成了一个组合的矩阵运算