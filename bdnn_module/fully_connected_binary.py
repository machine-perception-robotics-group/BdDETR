import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
import bdnn_module.binary_functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.autograd import Function
from bdnn_module.utils import decomposition as pp

class Linear_binary_Function(Function):

    @staticmethod
    def forward(ctx, input, weight, weight_approx, offset_W, M_ui, c, quantize_bits, basis, train_b, train_epoch, wa_mode, mode, num_threads, bias=None):
        output, weight_approx, M_ui, c = F.linear_binary_KA(input, weight, weight_approx, offset_W, M_ui, c, quantize_bits, basis, train_b,
                                                         train_epoch, mode, num_threads, bias)
        if train_b:
            ctx.save_for_backward(input, weight_approx, M_ui, c, bias)
        return output, M_ui, c

    @staticmethod
    def backward(ctx, grad_output, grad_M_ui, grad_c):
        input, weight, M_ui, c, bias = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, None, None, M_ui, c, None, None, None, None, None, None, grad_bias


#  based on pytorch
# @weak_module
class Linear_binary(Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, quantize_bits, basis, bias=True, train=False, d_mode="exh", M_ui=None, c=None, flag=False, biaas=None):
        super(Linear_binary, self).__init__()

        if torch.is_tensor(in_features) or torch.is_tensor(out_features):
            self.in_features = in_features.shape[2]
            self.out_features = out_features.shape[0]
        else:
            self.in_features = in_features
            self.out_features = out_features

        self.quantize_bits = quantize_bits
        self.basis = basis
        
        if torch.is_tensor(out_features) or torch.is_tensor(in_features):
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.weight_approx = Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=False)
        else:
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.weight_approx = Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=False)

        self.mode = d_mode
        self.train_epoch = train
        self.dim_len = 64
        self.bitset = (self.in_features + (self.dim_len - 1)) // self.dim_len

        if torch.is_tensor(out_features) or torch.is_tensor(in_features):
            # deit
            self.weight_np = np.zeros((self.out_features, self.in_features), dtype=np.float64)
            self.M_ui = Parameter(torch.LongTensor(self.out_features, basis, self.bitset, device='cpu'), requires_grad=False)
            self.c = Parameter(torch.DoubleTensor(self.out_features, basis, device='cpu'), requires_grad=False)
            self.M_ui_np = np.zeros((self.out_features, basis, self.bitset), dtype=np.uint64)
            self.c_np = np.zeros((self.out_features, basis), dtype=np.float64)
        else:
            # detr
            self.weight_np = np.zeros((out_features, in_features), dtype=np.float64)
            self.M_ui = Parameter(torch.LongTensor(out_features, basis, self.bitset, device='cpu'), requires_grad=False)
            self.c = Parameter(torch.DoubleTensor(out_features, basis, device='cpu'), requires_grad=False)
            self.M_ui_np = np.zeros((out_features, basis, self.bitset), dtype=np.uint64)
            self.c_np = np.zeros((out_features, basis), dtype=np.float64)

        self.offset_W = None
        self.wa_mode = False
        self.init_Mc = True
        self.num_threads = 4
        if bias:
            if flag:
                self.bias = biaas
            else:
                self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_approx, a=math.sqrt(5))
        init.zeros_(self.M_ui)
        init.zeros_(self.c)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # @weak_script_method
    def forward(self, input, M_ui=None, c=None):
        if self.offset_W is None:
            self.weight_np = self.weight.cpu().clone().detach().numpy()
            self.weight_np = self.weight_np.astype(np.float64)
            offset_W = pp.offset_fc(self.in_features, self.weight_np)
            offset_W = offset_W.astype(np.float64)
            self.offset_W = offset_W
            
            if M_ui is not None:
                self.M_ui_np = M_ui
                self.c_np = c
          
        output_a, M_ui, c = Linear_binary_Function.apply(input, self.weight, self.weight_approx, self.offset_W, self.M_ui_np, self.c_np, self.quantize_bits,
                                                         self.basis, self.training, self.train_epoch, self.wa_mode, self.mode, self.num_threads, self.bias)
        return output_a


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, quantize_bits={}, basis={}, mode={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.quantize_bits, self.basis, self.mode
        )
