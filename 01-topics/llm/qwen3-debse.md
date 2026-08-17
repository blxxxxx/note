以qwen3为例

dense

#### tokenizer
> "hello world!"
> ["hello", " world", "!"]
> [21, 400, 10]
> [vector(4096), vector(4096), vector(4096)] #embdeding也是随机初始化的
- 字符串切词
    - 训练（基于规则） byte-level BPE（先把所有预料数据变成一个一个byte，然后按照相邻出现次数进行合并，合并后获得了一个新的2byte的基本单元，以此类推）
    - 推理 基于规则算法 （根据训练时产生的优先级进行检索按照优先级合并各个byte）
- 变成token idenx
- 变成向量 类似于有一个vocab_size(152k)*hidden_size(4096)的大矩阵
> 不存在空格问题byte-level本质对于任意字符串进行切割，肯定包括空格的编码，还可能有前导空格等编码

#### transformer block（Transformer Decoder Layer）* N
```
x = embedding(input_ids)

for layer in layers:
    # 1. Attention 前的 norm
    residual = x
    x_norm = RMSNorm(x)

    # 2. Self-Attention
    q = q_proj(x_norm) # 维度不变 4096
    k = k_proj(x_norm) # 维度变小 因为后面GQA的时候num_kv_heads更小，1024
    v = v_proj(x_norm) # 如上

    q = q.view(batch, seq_len, num_q_heads, head_dim).transpose(1, 2) # 32个head每个128维
    k = k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2) # 8个head每个128维
    v = v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    q = q_norm(q) # 也是RMSNorm 即128维里面独立的进行均方根+放缩的归一化
    k = k_norm(k)

    q, k = RoPE(q, k)

    k = repeat_kv(k)  # 8 heads -> 32 heads
    v = repeat_kv(v)  # 8 heads -> 32 heads

    attn_score = q @ k.transpose(-2, -1) / sqrt(head_dim)
    attn_score += causal_mask
    attn_prob = softmax(attn_score)
    attn_out = attn_prob @ v

    attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, hidden_size)
    attn_out = o_proj(attn_out)

    attn_out = o_proj(attn_out)

    # 3. 残差连接
    x = residual + attn_out

    # 4. MLP 前的 norm
    residual = x
    x_norm = RMSNorm(x)

    # 5. SwiGLU MLP
    gate = gate_proj(x_norm)
    up = up_proj(x_norm)
    hidden = SiLU(gate) * up
    mlp_out = down_proj(hidden)

    # 6. 残差连接
    x = residual + mlp_out

# 7. 最后的 norm
x = RMSNorm(x)

# 8. 投影回词表
logits = lm_head(x)
probs = softmax(logits / temperature)
```
- res残差
    - 一个transformer block有两个残差 可以理解为norm+attention是学习残差，norm+mlp也是学习残差
- RMSnorm
    - 每个token之间不相互影响是独立的，即对于每个token的4096个数字作归一化
        ```
        layernorm
        μ = mean(x)
        σ = sqrt(mean((x - μ)^2) + eps)
        y = (x - μ) / σ
        out = y * γ + β  
        通过均值和方差进行归一化，然后每个维度都进行缩放和平移，γ β和都是可学习的参数，也是4096维，qwen3_eps取1e-6，防止除0

        RMSnorm
        rms = sqrt(mean(x^2) + eps)
        y = x / rms
        out = y * γ
        每个维度除均方根，然后放缩
        ```
    -  k和v的norm也是rmsnorm
- RoPE
    - rope每两个维度一组进行旋转
- causal_mask
    - 右上角为负无穷，其余都是0，主对角线也是0，因为自己的q是会和自己的k乘算出权重的
- softmax
    ```
    softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
    ```
- SwiGLU mlp
    ```
    普通mlp
    up = up_proj(x)
    hidden = SiLU(up) # 引入非线性
    out = down_proj(hidden)

    SwiGLU mlp
    up = up_proj(x)
    gate = gate_proj(x) # 学习应该放大哪些，遏制哪些
    hidden = SiLU(gate) * up # 逐元素相乘
    out = down_proj(hidden)

    所有proj都是可学习参数

    SiLU(x) = x * sigmoid(x) 
    sigmoid(x) = 1 / (1 + e^(-x))
    ```
- 所有可学习参数的矩阵乘法都no bias
- 有些lmhead和前面的embeddings用同一个矩阵，有些不是，qwen3不是
- 最后probs采样到token
    - greedy贪心
    - 概率（可以理解为生成一个0-1的随机数看看落在哪个区间里面）
        - topp 前缀概率
        - topk 前k和token
        - beamsearch
- 从离散的token变回字符串直接concat在一起就好了，因为所有字符都被编码了（byte-level BPE）