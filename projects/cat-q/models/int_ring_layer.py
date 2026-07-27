"""Checkpoint-compatible Bailing/Ring layer wrappers."""

import math
from typing import Optional, Tuple, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from transformers.models.llama.modeling_llama import (
    # apply_rotary_pos_emb,
    repeat_kv,
)

# ---- Project quantization primitives (must exist in your repo) ----
from quantize.int_linear_lora import LoRAQuantLinear,LoRABailingMoeV2Gate
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.catq_norm import CatQLlamaRMSNorm


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed

# -----------------------------
# Quant Attention (packed QKV)
# -----------------------------
class QuantBailingMoeV2Attention(nn.Module):
    """
    BailingMoeV2 packed-QKV attention:
      query_key_value: Linear(hidden, (n_heads + 2*n_kv_heads)*head_dim)
      dense:           Linear(n_heads*head_dim, hidden)
    """

    def __init__(
        self,
        org_module: nn.Module,
        config,
        args=None,
        use_lora=False,
        lora_attr=None,
        layer_id = None,
    ):
        super().__init__()

        assert use_lora==True
        self.config = config
        self.layer_idx = layer_id
        self.lora_attr = lora_attr

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.attn_dropout_p = float(getattr(config, "attention_dropout", 0.0))
        self.args = args


        # Quantized projections
        self.query_key_value = LoRAQuantLinear(
            org_module.query_key_value,
            args.weight_quant_params,
            args.act_quant_params,
            r=args.r,
            lora_attr=self.lora_attr
        )
        self.dense = LoRAQuantLinear(
            org_module.dense,
            args.weight_quant_params,
            args.act_quant_params,
            r=args.r,
            lora_attr=self.lora_attr
        )

        # Optional qk norm (kept in fp16)
        self.use_qk_norm = bool(getattr(config, "use_qk_norm", False))
        if self.use_qk_norm:
            # Reuse original norms if present; else create RMSNorm over head_dim
            self.query_layernorm = getattr(org_module, "query_layernorm", nn.Identity())
            self.key_layernorm = getattr(org_module, "key_layernorm", nn.Identity())

        # Rotary embedding (fallback if upstream doesn't pass position_embeddings)
        # self.rotary_emb = copy.deepcopy(org_module.rotary_emb)

        # Quantized matmul
        self.qkt_matmul = QuantMatMul(args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul)
        self.pv_matmul = QuantMatMul(args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,  # supports HF Cache
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        bsz, q_len, _ = hidden_states.size()

        # ---- packed QKV ----
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(
            bsz,
            q_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads],
            dim=-2,
        )
        # [bsz, heads, seq, dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        # ---- RoPE ----

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ---- cache update ----
        # HF Cache API: past_key_value.get_usable_length / update(layer_idx, key, value, cache_kwargs)
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ---- repeat kv (GQA) ----
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---- attention mask ----
        # If attention_mask is None, build causal mask (including cached kv_len if applicable).

        # user-provided additive mask should already incorporate causality if desired.
        attn_mask = attention_mask

        # ---- attn weights ----
        # [bsz, heads, q, kv]
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.training and self.attn_dropout_p > 0:
            attn_probs = F.dropout(attn_probs, p=self.attn_dropout_p)

        # ---- attn output ----
        attn_output = self.pv_matmul(attn_probs, value_states)  # [bsz, heads, q, dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# -----------------------------
# Quant MLP (dense)
# -----------------------------
class QuantBailingMoeV2MLP(nn.Module):
    """
    Standard gate/up/down MLP:
      y = down( act(gate(x)) * up(x) )
    """

    def __init__(self,
            org_module: nn.Module,
            hidden_act: str,
            args: Any,
            use_lora=False,
            lora_attr=None,
            ):
        super().__init__()
        # self.act = _get_activation_fn(getattr(config, "hidden_act", "silu"))
        self.act_fn = ACT2FN[hidden_act]
        self.lora_attr = lora_attr

        assert use_lora == True

        self.gate_proj = LoRAQuantLinear(
            org_module.gate_proj,
            args.weight_quant_params,
            args.act_quant_params,
            r=args.r,
            lora_attr=self.lora_attr
        )
        self.up_proj = LoRAQuantLinear(
            org_module.up_proj,
            args.weight_quant_params,
            args.act_quant_params,
            r=args.r,
            lora_attr=self.lora_attr
        )
        self.down_proj = LoRAQuantLinear(
            org_module.down_proj,
            args.weight_quant_params,
            args.act_quant_params,
            r=args.r,
            lora_attr=self.lora_attr
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# -----------------------------
# Quant Sparse MoE wrapper
# -----------------------------
def get_quant_moe_mlp(org_module: nn.Module,args,config,use_lora,lora_attr):


    if use_lora:
        weight_quant_params = copy.deepcopy(args.weight_quant_params)
        act_quant_params = copy.deepcopy(args.act_quant_params)

        if args.only_lora_gate is True:
            weight_quant_params["n_bits"] = 16
            act_quant_params["n_bits"] = 16

        org_module.gate  = LoRABailingMoeV2Gate(
                org_module.gate,
                weight_quant_params,
                act_quant_params,
                r=args.r,
                lora_attr=lora_attr
            )
    else:
        org_module.gate  = QuantLinear(
                org_module.gate,
                args.weight_quant_params,
                args.act_quant_params,
            )


    if config.num_shared_experts is not None:
        org_module.shared_experts = QuantBailingMoeV2MLP(
            org_module=org_module.shared_experts,
            hidden_act=config.hidden_act,
            args=args,
            use_lora=use_lora,
            lora_attr=lora_attr
        )


    for i in range(len(org_module.experts)):
        org_module.experts[i] = QuantBailingMoeV2MLP(
            org_module=org_module.experts[i],
            hidden_act=config.hidden_act,
            args=args,
            use_lora=use_lora,
            lora_attr=lora_attr
        )
    return org_module


# -----------------------------
# Decoder Layer (interface compatible)
# -----------------------------
class QuantBailingMoeV2DecoderLayer(nn.Module):
    """
    Interface-compatible with your QuantLlamaDecoderLayer forward signature.

    Expected original layer fields:
      - attention: attention module with query_key_value + dense
      - mlp: either dense MLP (gate/up/down) or sparse MoE block
      - input_layernorm, post_attention_layernorm
    """

    def __init__(
        self,
        config,
        ori_layer: nn.Module,
        layer_id: int,
        args: Any,
        quant_mode="fp16",
        use_lora=False,
        lora_attr=None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.lora_attr = lora_attr
        self.use_lora = use_lora
        assert self.use_lora == True
        self.quant_mode = quant_mode
        self.args = args

        assert self.quant_mode in ["fp16", "weight_merge"]
        self.finished_quant = False


        # Norms (use CatQLlamaRMSNorm wrapper in your project)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.input_layernorm = CatQLlamaRMSNorm(ori_layer.input_layernorm, eps=eps)
        self.post_attention_layernorm = CatQLlamaRMSNorm(ori_layer.post_attention_layernorm, eps=eps)

        # Attention (packed qkv)
        self.attention = QuantBailingMoeV2Attention(
            org_module=ori_layer.attention,
            config=config,
            args=args,
            layer_id=layer_id,
            use_lora=self.use_lora,
            lora_attr=self.lora_attr,
        )

        # MLP / MoE
        mlp = ori_layer.mlp
        if hasattr(mlp, "experts") and isinstance(mlp.experts, nn.ModuleList):
            self.mlp = get_quant_moe_mlp(
                org_module=ori_layer.mlp,
                args=args,
                config=config,
                use_lora=self.use_lora,
                lora_attr=self.lora_attr
            )
        else:
            self.mlp = QuantBailingMoeV2MLP(
                org_module=ori_layer.mlp,
                hidden_act=config.hidden_act,
                args=args,
                use_lora=self.use_lora,
                lora_attr=self.lora_attr
            )

        self.update_quant_mode(self.quant_mode,args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # ---- Attention block ----
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out, attn_weights, present_kv = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_out

        # ---- MLP / MoE block ----
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        router_logits = None
        mlp_out = self.mlp(hidden_states)
        if isinstance(mlp_out, tuple):
            hidden_states = mlp_out[0]
            router_logits = mlp_out[1]
            # Bailing may return (router_logits, topk_idx)
            if isinstance(router_logits, tuple):
                router_logits = router_logits[0]
        else:
            hidden_states = mlp_out

        hidden_states = residual + hidden_states

        outputs: Tuple[Any, ...] = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_kv,)
        if output_router_logits:
            outputs += (router_logits,)

        return outputs

    def set_quant_state(self, weight_quant=False, act_quant=False, quant_rate=1.0):
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant, quant_rate)


    def update_quant_mode(self, new_mode, args=None):
        self.quant_mode = new_mode
        support_list = ["fp16", "weight_merge"]
        if self.quant_mode not in support_list:
            raise ValueError(f"Only inference modes are supported: {support_list}")

        if self.quant_mode == "fp16":
            self.set_quant_state(False, False, args.quant_rate)
            return

        self.set_quant_state(False, args.abits < 16, args.quant_rate)
        if self.finished_quant:
            return
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, LoRAQuantLinear):
                    module.merged = True
                    weight = module.weight
                    if args.r > 0:
                        for index in range(module.lora_iter_num):
                            weight = weight + module.lora_B[index] @ module.lora_A[index] * module.scaling
                    module.weight = module.weight_quantizer(weight)
                elif isinstance(module, QuantLinear):
                    module.weight = module.weight_quantizer(module.weight)
        self.finished_quant = True
