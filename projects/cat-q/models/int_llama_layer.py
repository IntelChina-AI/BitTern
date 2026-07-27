import torch
from torch import nn
from typing import Optional, Tuple
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
import torch.nn.functional as F
from quantize.catq_norm import CatQLlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb,repeat_kv
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.activations import ACT2FN
import copy
from models.transformation import (
    smooth_fc_fc_inplace,
    smooth_ln_fcs_inplace,
    smooth_q_k_inplace,
    truncate_number,
)

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from quantize.int_linear_lora import LoRAQuantLinear

class QuantLlamaMLP(nn.Module):
    def __init__(
        self,
        org_module: nn.Module,
        hidden_act: str,
        args=None,
        use_lora=False,
        lora_attr=None,
    ):
        super().__init__()
        # self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.lora_attr = lora_attr
        self.args = args
        if use_lora:
            self.gate_proj = LoRAQuantLinear(
                org_module.gate_proj,
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
            self.up_proj = LoRAQuantLinear(
                org_module.up_proj,
                args.weight_quant_params,
                args.act_quant_params,
                r=args.r,
                lora_attr=self.lora_attr
            )
        else:
            self.gate_proj = QuantLinear(
                org_module.gate_proj,
                args.weight_quant_params,
                args.act_quant_params,
            )
            self.down_proj = QuantLinear(
                org_module.down_proj,
                args.weight_quant_params,
                args.act_quant_params,
            )
            self.up_proj = QuantLinear(
                org_module.up_proj, args.weight_quant_params, args.act_quant_params
            )
        self.act_fn = ACT2FN[hidden_act]



    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class QuantQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, org_module: nn.Module, args, config, use_lora, lora_attr):
        super().__init__()
        self.args = args
        self.num_experts = org_module.num_experts
        self.top_k = org_module.top_k
        self.norm_topk_prob = org_module.norm_topk_prob
        self.is_quant_moe_block = True

        if args.quant_gate is True or args.only_lora_gate is True:
            if use_lora:
                weight_quant_params = copy.deepcopy(args.weight_quant_params)
                act_quant_params = copy.deepcopy(args.act_quant_params)

                if args.only_lora_gate is True:
                    weight_quant_params["n_bits"] = 16
                    act_quant_params["n_bits"] = 16

                self.gate = LoRAQuantLinear(
                    org_module.gate,
                    weight_quant_params,
                    act_quant_params,
                    r=args.r,
                    lora_attr=lora_attr,
                )
            else:
                self.gate = QuantLinear(
                    org_module.gate,
                    args.weight_quant_params,
                    args.act_quant_params,
                )
        else:
            self.gate = org_module.gate

        self.experts = nn.ModuleList(
            [
                QuantLlamaMLP(
                    org_module=org_module.experts[i],
                    hidden_act=config.hidden_act,
                    args=args,
                    use_lora=use_lora,
                    lora_attr=lora_attr,
                )
                for i in range(len(org_module.experts))
            ]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[top_x]
            expert_down_output = expert_layer(current_state)
            current_hidden_states = expert_down_output * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def get_quant_moe_mlp(org_module: nn.Module,args,config,use_lora,lora_attr):
    return QuantQwen3MoeSparseMoeBlock(
        org_module=org_module,
        args=args,
        config=config,
        use_lora=use_lora,
        lora_attr=lora_attr,
    )


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self,
                 org_module: nn.Module,
                 config: LlamaConfig,
                 args=None,
                 use_lora=False,
                 lora_attr=None,
                 layer_id = None,
                ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.lora_attr = lora_attr
        self.layer_idx = layer_id
        self.scaling = self.head_dim**-0.5
        self.add_norm_model_list = ["qwen3" ,"qwen3_moe"]

        if config.model_type in self.add_norm_model_list :
            self.q_norm = copy.deepcopy(org_module.q_norm)
            self.k_norm = copy.deepcopy(org_module.k_norm)

        self.args = args
        if (self.head_dim * self.num_heads) != self.hidden_size and self.config.model_type not in self.add_norm_model_list:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # self.rotary_emb = copy.deepcopy(org_module.rotary_emb)
        self.rotary_emb = LlamaRotaryEmbedding(self.config)

        if use_lora:
            self.k_proj = LoRAQuantLinear(
                org_module.k_proj,
                args.weight_quant_params,
                args.act_quant_params,
                disable_input_quant=False,
                r=args.r,
                lora_attr=self.lora_attr
            )
            self.v_proj = LoRAQuantLinear(
                org_module.v_proj,
                args.weight_quant_params,
                args.act_quant_params,
                disable_input_quant=False,
                r=args.r,
                lora_attr=self.lora_attr
            )
            self.q_proj = LoRAQuantLinear(
                org_module.q_proj,
                args.weight_quant_params,
                args.act_quant_params,
                disable_input_quant=False,
                r=args.r,
                lora_attr=self.lora_attr
            )
            self.o_proj = LoRAQuantLinear(
                org_module.o_proj,
                args.weight_quant_params,
                args.act_quant_params,
                r=args.r,
                lora_attr=self.lora_attr
            )
        else:
            self.k_proj = QuantLinear(
                org_module.k_proj,
                args.weight_quant_params,
                args.act_quant_params,
                disable_input_quant=False,
            )
            self.v_proj = QuantLinear(
                org_module.v_proj,
                args.weight_quant_params,
                args.act_quant_params,
                disable_input_quant=False,
            )
            self.q_proj = QuantLinear(
                org_module.q_proj,
                args.weight_quant_params,
                args.act_quant_params,
                disable_input_quant=False,
            )
            self.o_proj = QuantLinear(
                org_module.o_proj, args.weight_quant_params, args.act_quant_params
            )

        self.qkt_matmul = QuantMatMul(
            args.q_quant_params, args.k_quant_params, matmul_func=torch.matmul
        )
        self.pv_matmul = QuantMatMul(
            args.p_quant_params, args.v_quant_params, matmul_func=torch.matmul
        )



    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.config.model_type in self.add_norm_model_list:
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)



        kv_seq_len = key_states.shape[-2]

        if past_key_value is not None:
            # kv_seq_len += past_key_value[0].shape[-2]
            if self.layer_idx is None:
                    raise ValueError(
                        f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                        "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                        "with a layer index."
                    )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # cos, sin = self.rotary_emb(value_states,position_ids)  # for transformer > 4.37.2
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)



        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)






        if self.args.disable_quant is False:
            query_states = self.qkt_matmul.quant_x1(query_states) # dont quant q
        key_states = self.qkt_matmul.quant_x2(key_states)
        attn_weights = self.qkt_matmul(query_states, key_states.transpose(2, 3)) * self.scaling


        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        if self.args.disable_quant is False:
            attn_weights = self.pv_matmul.quant_x1(attn_weights) # dont quant p
        value_states = self.pv_matmul.quant_x2(value_states)
        attn_output = self.pv_matmul(attn_weights, value_states)


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 ori_layer,
                 layer_id,
                 args,
                 quant_mode="fp16",
                 use_lora=False,
                 lora_attr=None,
                 ):
        super().__init__()
        self.use_lora = use_lora
        self.hidden_size = config.hidden_size
        self.lora_attr = lora_attr

        self.self_attn = QuantLlamaAttention(
            org_module=ori_layer.self_attn,
            config=config,
            args=args,
            use_lora=self.use_lora,
            lora_attr=self.lora_attr,
            layer_id = layer_id,
            )
        if isinstance(ori_layer.mlp, Qwen3MoeSparseMoeBlock):
            self.mlp = get_quant_moe_mlp(
                org_module=ori_layer.mlp,
                args=args,
                config=config,
                use_lora=self.use_lora,
                lora_attr=self.lora_attr
            )
        else:
            self.mlp = QuantLlamaMLP(
                org_module=ori_layer.mlp,
                hidden_act=config.hidden_act,
                args=args,
                use_lora=self.use_lora,
                lora_attr=self.lora_attr
            )

        self.input_layernorm = CatQLlamaRMSNorm(ori_layer.input_layernorm,eps=ori_layer.input_layernorm.variance_epsilon)
        self.post_attention_layernorm = CatQLlamaRMSNorm(ori_layer.post_attention_layernorm,eps=ori_layer.post_attention_layernorm.variance_epsilon)
        self.quant_mode = quant_mode

        assert self.quant_mode in ["fp16", "weight_merge"]
        self.finished_quant = False
        self.args = args
        self.update_quant_mode(self.quant_mode, args)


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
        if args.use_scaling:
            self.smooth_inplace()
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_router_logits = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)


        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,

        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)


        hidden_states = self.mlp(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
        return outputs

    def set_quant_state(self, weight_quant=False, act_quant=False, quant_rate=1.0):
        for m in self.modules():
            if isinstance(m, (QuantLinear, QuantMatMul)):
                m.set_quant_state(weight_quant, act_quant, quant_rate)

    @torch.no_grad()
    def smooth_inplace(self):
        for name, module in self.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        smooth_ln_fcs_inplace(
            self.input_layernorm,
            [self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj],
            self.qkv_smooth_scale,
            self.qkv_smooth_shift,
        )
        smooth_ln_fcs_inplace(
            self.post_attention_layernorm,
            [self.mlp.up_proj, self.mlp.gate_proj],
            self.fc1_smooth_scale,
            self.fc1_smooth_shift,
        )
        smooth_fc_fc_inplace(
            self.self_attn.v_proj,
            self.self_attn.o_proj,
            self.out_smooth_scale,
            self.out_smooth_shift,
            self.self_attn.num_key_value_groups,
            head_dim=self.self_attn.head_dim,
            args=self.args,
        )
        smooth_q_k_inplace(
            self.self_attn.q_proj,
            self.self_attn.k_proj,
            self.qkt_smooth_scale,
            self.self_attn.num_key_value_groups,
            head_dim=self.self_attn.head_dim,
            args=self.args,
        )
        if self.args.use_down_scale:
            smooth_fc_fc_inplace(
                self.mlp.up_proj,
                self.mlp.down_proj,
                self.fc2_smooth_scale,
                self.fc2_smooth_shift,
            )
