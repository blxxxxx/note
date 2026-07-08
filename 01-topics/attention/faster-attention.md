qwen3-next & qwen3.5
3*linear attention + full attention
线性注意力用kv来维护一个记忆矩阵 q与记忆矩阵乘得到最终结果
gate：门控？

deepseekv4
也是两种交替
1 较轻压缩kv，窗口不压缩，检索窗口外的
2 较重压缩kv，全注意力

一些通用的
- GQA