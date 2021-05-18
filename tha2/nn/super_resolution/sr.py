from typing import List, Union

from torch import Tensor
from tha2.nn.batch_module.batch_input_module import BatchInputModule, BatchInputModuleFactory

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(layer_func: nn.Module, num_layers: int, **kwarg) -> nn.Sequential:
    """Make layers by stacking the same blocks.

    Args:
        layer_func (nn.module): nn.module class for basic block.
        num_layers (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_layers):
        layers.append(layer_func(**kwarg))
    return nn.Sequential(*layers)


def default_init_weights(module_list: Union[List[nn.Module], nn.Module], scale: int = 1, bias_fill: int = 0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features. Default: 64.
        res_scale (float): Residual scale. Default: 1.
        torch_init (bool): If set to True, use pytorch default init,
                           otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, torch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not torch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class MSRResNet(BatchInputModule):
    """Modified SRResNet.
    A compacted version modified from SRResNet in
      "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4.
            Default: 4.
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_block=16,
                 upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        # up-sampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat,
                                     num_feat * self.upscale * self.upscale, 3,
                                     1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out

    def forward_from_batch(self, batch: List[Tensor]) -> Tensor:
        return self.forward(batch[0])


class SRFactory(BatchInputModuleFactory):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=16,
                 num_block=4,
                 upscale=2):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_block = num_block
        self.upscale = upscale

    def create(self) -> BatchInputModule:
        return MSRResNet(self.num_in_ch, self.num_out_ch, self.num_feat, self.num_block, self.upscale)


if __name__ == "__main__":
    cuda = torch.device('cuda')
    sr = SRFactory().create().to(cuda)

    image = torch.randn(8, 3, 256, 256, device=cuda)
    outputs = sr(image)
    for i in range(len(outputs)):
        print(i, outputs[i].shape)
