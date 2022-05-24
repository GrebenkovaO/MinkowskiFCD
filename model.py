# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from examples.resnet import ResNetBase

class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        
        self.W_gate = nn.Sequential(
            ME.MinkowskiConvolution(F_g, n_coefficients, kernel_size=1, stride=1, bias=True, dimension = 3),
            ME.MinkowskiBatchNorm(n_coefficients)
        )
        
        self.W_x = nn.Sequential(
            ME.MinkowskiConvolution(F_l, n_coefficients, kernel_size=1, stride=1, bias=True, dimension = 3),
            ME.MinkowskiBatchNorm(n_coefficients)
        )

        self.psi = nn.Sequential(
            ME.MinkowskiConvolution(n_coefficients, 1, kernel_size=1, stride=1, bias=True, dimension = 3),
            ME.MinkowskiBatchNorm(1),
            ME.MinkowskiSigmoid()
        )

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        #print(gate.shape, 'gate')
        g1 = self.W_gate(gate)
        #print(g1.shape, 'g1')
        #print(skip_connection.shape, 'skip_connection')
        x1 = self.W_x(skip_connection)
        #print(x1.shape, 'x1')
        psi = self.relu(g1 + x1)
        #print(psi.shape, 'psi1')
        psi = self.psi(psi)
        #print(psi.shape, 'psi2')
        out = skip_connection * psi
        return out


class IdentityAttention(nn.Module):
    def __init__(self):
        super(IdentityAttention, self).__init__()

    def forward(self, gate, skip_connection):
        return gate


class MinkUNetBaseAttention(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3, attention=False):
        self.attention = attention
        ResNetBase.__init__(self, in_channels, out_channels, D)
        print('MinkUNet' + 'with attention'*self.attention) 

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM

        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        
        if self.attention:
            self.Att0 = AttentionBlock(F_g=256, F_l=128, n_coefficients=128)
        else:
            self.Att0 = IdentityAttention()

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])
        
        if self.attention:
            self.Att1 = AttentionBlock(F_g=128, F_l=64, n_coefficients=64)
        else:
            self.Att1 = IdentityAttention()

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])
        
        if self.attention:
            self.Att2 = AttentionBlock(F_g=96, F_l=32, n_coefficients=32)
        else:
            self.Att2 = IdentityAttention()

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])
        
        if self.attention:
            self.Att3 = AttentionBlock(F_g=96, F_l=32, n_coefficients=32)
        else:
            self.Att3 = IdentityAttention()

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)

        att0 = self.Att0(gate=out, skip_connection=out_b3p8)
        out = ME.cat(att0, out)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        
        att1 = self.Att1(gate=out, skip_connection=out_b2p4)
        out = ME.cat(att1, out)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        
        att2 = self.Att2(gate=out, skip_connection=out_b1p2)
        out = ME.cat(att2, out)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        
        att3 = self.Att3(gate=out, skip_connection=out_p1)
        out = ME.cat(att3, out)
        out = self.block8(out)

        return self.final(out)



class MinkUNet34Attention(MinkUNetBaseAttention):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class MinkUNet34CAttention(MinkUNet34Attention):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


# if __name__ == '__main__':
#     from tests.python.common import data_loader
#     # loss and network
#     criterion = nn.CrossEntropyLoss()
#     net = MinkUNet14A(in_channels=3, out_channels=5, D=2)
#     print(net)

#     # a data loader must return a tuple of coords, features, and labels.
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     net = net.to(device)
#     optimizer = SGD(net.parameters(), lr=1e-2)

#     for i in range(10):
#         optimizer.zero_grad()

#         # Get new data
#         coords, feat, label = data_loader(is_classification=False)
#         input = ME.SparseTensor(feat, coordinates=coords, device=device)
#         label = label.to(device)

#         # Forward
#         output = net(input)

#         # Loss
#         loss = criterion(output.F, label)
#         print('Iteration: ', i, ', Loss: ', loss.item())

#         # Gradient
#         loss.backward()
#         optimizer.step()

#     # Saving and loading a network
#     torch.save(net.state_dict(), 'test.pth')
#     net.load_state_dict(torch.load('test.pth'))
