"""LoRA-aware linear wrappers used to restore and merge released parameters."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize.int_linear import QuantLinear
from quantize.quantizer import TernaryQuantizer, UniformAffineQuantizer


class LoRAQuantLinear(QuantLinear):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params,
        act_quant_params,
        disable_input_quant=False,
        r=0,
        lora_alpha=1,
        lora_attr=None,
    ):
        super().__init__(
            org_module,
            weight_quant_params,
            act_quant_params,
            disable_input_quant,
        )
        lora_attr = lora_attr or {"lora_iter_num": 1}
        self.r = r
        self.lora_iter_num = lora_attr["lora_iter_num"]
        self.merged = False

        if r > 0:
            out_features, in_features = self.weight.shape
            self.lora_A = nn.ParameterList(
                [
                    nn.Parameter(
                        self.weight.new_zeros((r, in_features)),
                        requires_grad=False,
                    )
                    for _ in range(self.lora_iter_num)
                ]
            )
            self.lora_B = nn.ParameterList(
                [
                    nn.Parameter(
                        self.weight.new_zeros((out_features, r)),
                        requires_grad=False,
                    )
                    for _ in range(self.lora_iter_num)
                ]
            )
            self.scaling = lora_alpha / r

        if weight_quant_params["n_bits"] == 1:
            self.weight_quantizer = TernaryQuantizer(
                weight_quant_params=weight_quant_params,
                shape=self.weight.shape,
            )
            self.act_quantizer = (
                None
                if disable_input_quant
                else UniformAffineQuantizer(**act_quant_params)
            )

    def forward(self, inputs):
        weight = self.weight
        bias = self.bias
        if weight.device != inputs.device:
            weight = weight.to(inputs.device)
            if bias is not None:
                bias = bias.to(inputs.device)

        if not self.merged and self.use_weight_quant:
            if self.r > 0:
                weight = weight + self.lora_B[0] @ self.lora_A[0] * self.scaling
            weight = self.weight_quantizer(weight, self.quant_rate)
        if self.use_act_quant and not self.disable_input_quant:
            inputs = self.act_quantizer(inputs, self.quant_rate)
        return self.fwd_func(inputs, weight, bias, **self.fwd_kwargs)


class LoRABailingMoeV2Gate(LoRAQuantLinear):
    def __init__(self, org_module, **kwargs):
        super().__init__(org_module=org_module, **kwargs)
        self.top_k = org_module.top_k
        self.num_experts = org_module.num_experts
        self.n_group = org_module.n_group
        self.topk_group = org_module.topk_group
        self.routed_scaling_factor = org_module.routed_scaling_factor
        self.register_buffer("expert_bias", org_module.expert_bias)

    def group_limited_topk(self, scores):
        num_tokens = scores.shape[0]
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_indices = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_indices, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked_scores, k=self.top_k, dim=-1)

    def forward(self, hidden_states):
        weight = self.weight
        if not self.merged and self.use_weight_quant:
            if self.r > 0:
                weight = weight + self.lora_B[0] @ self.lora_A[0] * self.scaling
            weight = self.weight_quantizer(weight, self.quant_rate)
        if self.use_act_quant and not self.disable_input_quant:
            hidden_states = self.act_quantizer(hidden_states, self.quant_rate)

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.float(), weight.float(), self.bias)
        scores = torch.sigmoid(logits.float()).type_as(logits)
        routing_scores = scores + self.expert_bias
        _, topk_indices = self.group_limited_topk(routing_scores)
        topk_weight = torch.gather(scores, dim=1, index=topk_indices).type_as(logits)
        if self.top_k > 1:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_indices, topk_weight, logits
