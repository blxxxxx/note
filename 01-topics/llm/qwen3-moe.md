

```
residual = x
x = RMSNorm(x) # RMSNorm over hidden_size
x = qwen3_moe_block(x)
x = residual + x

def qwen3_moe_block(x):
    # x: [batch, seq_len, hidden_size]
    b, s, h = x.shape
    x_flat = x.view(b * s, h)

    # router router_logits: [b * s, num_experts]
    router_logits = x_flat @ router_weight.T
    router_probs = softmax(router_logits, dim=-1, dtype=float32)

    topk_weights, topk_experts = topk(router_probs, k=8, dim=-1)

    # Qwen3-MoE config: norm_topk_prob = true
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    final = zeros_like(x_flat)

    for expert_id in used_experts(topk_experts):
        token_idx, slot_idx = tokens_that_selected(expert_id)

        x_e = x_flat[token_idx]

        # fused gate/up projection
        gate_up = x_e @ gate_up_proj[expert_id].T
        gate, up = chunk(gate_up, 2, dim=-1)

        # SwiGLU
        hidden = SiLU(gate) * up

        # down projection
        out = hidden @ down_proj[expert_id].T

        # router weighted sum
        out = out * topk_weights[token_idx, slot_idx, None]

        # accumulate back to token positions
        final.index_add_(0, token_idx, out)

    return final.view(b, s, h)
```


router_logits = x @ router_weight.T
router_probs = softmax(router_logits)
topk_probs, topk_experts = topk(router_probs, k=8)




没有强制 shared expert
没有 token 必须走某个 expert
有 top-8 动态路由
有 load balancing 辅助损失
没有 DeepSeek 那套 shared + routed 的复杂 MoE 结构