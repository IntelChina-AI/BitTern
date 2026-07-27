"""RMSNorm wrapper used while reconstructing released checkpoint layer names."""

import torch
import torch.nn as nn


class CatQLlamaRMSNorm(nn.Module):
    def __init__(self, original, eps=1e-6):
        super().__init__()
        self.register_buffer("weight", original.weight)
        self.bias = None
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * normalized
        if self.bias is not None:
            output = output + self.bias
        return output.to(input_dtype)
