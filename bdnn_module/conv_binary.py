import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import bdnn_module.binary_functional as bF
from torch.nn import init
from torch.nn.modules.module import Module
from torch.autograd import Function
from torch.nn.modules.utils import _single, _pair, _triple
from bdnn_module.utils import decomposition as pp


class Conv2d_binary_Function(Function):

    @staticmethod
    def forward(ctx, input, weight, offset_W, M_ui, c, stride, padding, dilation, groups, quantize_bits, basis,
                train_b, train_epoch, d_mode, bias=None):
        output, weight_approx, M_ui, c = bF.conv2d_binary(input, weight, offset_W, M_ui, c, stride, padding, dilation, groups,
                                                          quantize_bits, basis, train_b, train_epoch, d_mode, bias)

        param_stdg = torch.tensor([*stride, *padding, *dilation, groups], requires_grad=False)
        if train_b:
            ctx.save_for_backward(input, weight_approx, M_ui, c, bias, param_stdg)
        if bias is not None:
            output = bias.view(-1, 1, 1) + output

        return output, M_ui, c

    @staticmethod
    def backward(ctx, grad_output, grad_M_ui, grad_c):
        input, weight, M_ui, c, bias, params_stdg = ctx.saved_tensors

        stride = (params_stdg[0].item(), params_stdg[1].item())
        padding = (params_stdg[2].item(), params_stdg[3].item())
        dilation = (params_stdg[4].item(), params_stdg[5].item())
        groups = params_stdg[6].item()
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = F.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = F.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, None, M_ui, c, None, None, None, None, None, None, None, None, None, grad_bias


# @weak_module
class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, quantize_bits, basis, bias, padding_mode, train, d_mode):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.quantize_bits = quantize_bits
        self.basis = basis
        self.basis_b = basis
        self.train_epoch = train
        self.mode = d_mode
        self.init_Mc = True
        self.offset_W = None
        self.num_threads = 4

        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.dim_len = 64
        self.in_size = in_channels * self.weight.shape[2] * self.weight.shape[3]
        self.bitset = (self.in_size + (self.dim_len - 1)) // self.dim_len
        self.weight_np = np.zeros((out_channels, in_channels, *kernel_size), dtype=np.float64)
        self.M_ui = Parameter(torch.LongTensor(out_channels, basis, self.bitset, device='cpu'), requires_grad=False)
        self.c = Parameter(torch.DoubleTensor(out_channels, basis, device='cpu'), requires_grad=False)
        self.M_ui_np = np.zeros((out_channels, basis, self.bitset), dtype=np.uint64)
        self.c_np = np.zeros((out_channels, basis), dtype=np.float64)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, quantize_bits={quantize_bits}, basis={basis}, mode={mode}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


#  based on pytorch
# @weak_module
class Conv2d_binary(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, quantize_bits=1, basis=1,
                 bias=True, padding_mode='zeros', train=True, d_mode='exh'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d_binary, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, quantize_bits, basis, bias, padding_mode, train, d_mode)


    # @weak_script_method
    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output_a, M_ui, c = Conv2d_binary_Function.apply(F.pad(input, expanded_padding, mode='circular'),
                            self.weight, self.M_ui, self.c, self.stride,
                            _pair(0), self.dilation, self.groups, self.quantize_bits, self.basis, self.training,
                            self.train_epoch, self.mode, self.bias)
            return output_a
        if self.offset_W is None:  # generate offset
            weight_np = self.weight.cpu().clone().detach().numpy()
            self.weight_np = weight_np.astype(np.float64)
            in_size = input.shape[1] * self.weight_np.shape[2] * self.weight_np.shape[3]
            self.offset_W = pp.offset(in_size, self.weight_np)
        output_a, weight_approx, M_ui, c = bF.conv2d_binary(input, self.weight_np, self.offset_W, self.M_ui_np, self.c_np, self.stride,
                                        self.padding, self.dilation, self.groups, self.quantize_bits, self.basis, self.training,
                                        self.train_epoch, self.mode, self.num_threads, self.bias)
        return output_a
