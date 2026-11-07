import torch
from torch import nn

class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x >= bound) | (grad_output < 0)
        return pass_through_if * grad_output, None


class LowerBound(nn.Module):
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))
    
    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return LowerBoundFunction.apply(x, self.bound)
