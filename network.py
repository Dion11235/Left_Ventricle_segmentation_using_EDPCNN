# https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
import torch
from torch import nn
import torch.nn.functional as F


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class UNet(nn.Module):
    def __init__(self, in_dim=1, out_dim=2, depth=5, wf=6, padding=False,
                 batch_norm=True, up_mode='upconv4', first_layer_pad=92, last_layer_resize=False):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv4', 'upconv2' or 'upsample'.
                           'upconv4'/'upconv2' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv4', 'upconv2', 'rsz_conv_nn3', 'rsz_conv_bl3', 'rsz_conv_bl1')
        self.padding = padding
        self.depth = depth
        prev_channels = in_dim
        if first_layer_pad is not None:
            self.first_layer_pad = nn.ConstantPad2d(first_layer_pad, 0.0)
        else:
            self.first_layer_pad = None
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, out_dim, kernel_size=1)
        self.last_layer_resize = last_layer_resize
        initialize_weights(self)

    def forward(self, x):
        in_size = x.size()[-2:]
        if self.first_layer_pad is not None:
            # print("")
            # print("size before padding : ",x.size())
            x = self.first_layer_pad(x)
            # print("size after padding : ",x.size())
            # print('')


        blocks = []
        # print("Down Path")
        # print("")
        for i, down in enumerate(self.down_path):
            # print("shape before maxpool and down : ", x.size())
            x = down(x)
            #print("shape before maxpool : ", x.size())
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)
                #print("shape after maxpool : ", x.size())
        # print('')
        # print("Up Path")
        # print("")
        for i, up in enumerate(self.up_path):
            #print("Before up : ", x.size())
            x = up(x, blocks[-i-1])
            #print("after up and crop and interpolate: ", x.size())

        x = self.last(x)
        #print("after last : ", x.size())
        if self.last_layer_resize:
            x = F.interpolate(x, in_size, mode='bilinear')
            #print("after interpolation : ", x.size())
            #print('')
            #print('')
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, momentum=0.99, eps=1e-3))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, momentum=0.99, eps=1e-3))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv4':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_size, momentum=0.99, eps=1e-3),
                nn.ReLU(),
            )

        elif up_mode == 'upconv2':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_size, momentum=0.99, eps=1e-3),
                nn.ReLU(),
            )
        elif up_mode == 'rsz_conv_nn3':
            self.up = nn.Sequential(nn.Upsample(mode='nearest', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        elif up_mode == 'rsz_conv_bl3':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        elif up_mode == 'rsz_conv_bl1':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        else:
            raise NotImplementedError

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()

        # print("")
        # print("layer aize : ",layer.size())
        # print("target size : ",target_size)
        # print("")

        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        #print("after up : ", up.size())
        crop1 = self.center_crop(bridge, up.shape[2:])
        #print("after crop : ", crop1.size())
        out = torch.cat([up, crop1], 1)
        #print("after Concat : ",out.size())
        out = self.conv_block(out)
        #print("after convolution : ", out.size())

        return out
