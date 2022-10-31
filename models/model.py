import torch.nn as nn
from models.functions import ReverseLayerF
import torch
from torchvision import models
import math
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'lse']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'lse'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        #if not no_spatial:
            #self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        #if not self.no_spatial:
            #x_out = self.SpatialGate(x_out)
        return x_out




class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        
        pre_a =  models.alexnet(pretrained=True)
        self.features = pre_a.features[0:8]
        for i,p in enumerate(self.features.parameters()):
            #if(i<6): #[changed to [2,4,6] depending on how many conv layers to freeze], 132 for mobilenet
            p.requires_grad = False

        
        self.bn1 = nn.BatchNorm2d(128) #128 for digit 256 for alpha
        self.bn2 = nn.BatchNorm2d(64) # 64 for digit 128 for alpha
        self.relu = nn.ReLU(True)

        self.cbam1 = CBAM(128) #128 for digit 256 for alpha
        self.cbam2 = CBAM(64) # 64 for digit 128 for alpha

        self.bottleneck2 = nn.Sequential()
        self.bottleneck2.add_module('b2_conv1',nn.Conv2d(128, 32, kernel_size=1,stride=1,padding = 'same')) # change with 128 and 32 for digit, 256 and 64 for alpha
        #self.bottleneck2.add_module('b2_r1',nn.ReLU(True))
        self.bottleneck2.add_module('b2_bn1',nn.BatchNorm2d(32))
        self.bottleneck2.add_module('b2_r1',nn.ReLU(True))
        self.bottleneck2.add_module('b2_conv2',nn.Conv2d(32, 32, kernel_size=3,stride=1,padding = 'same',dilation=2))
        #self.bottleneck2.add_module('b2_r2',nn.ReLU(True))
        self.bottleneck2.add_module('b2_bn2',nn.BatchNorm2d(32))
        self.bottleneck2.add_module('b2_r2',nn.ReLU(True))
        self.bottleneck2.add_module('b2_conv3',nn.Conv2d(32, 128, kernel_size=1,stride=1,padding = 'same'))
        #self.bottleneck2.add_module('b2_r3',nn.ReLU(True))
        self.bottleneck2.add_module('b2_bn3',nn.BatchNorm2d(128))
        self.bottleneck2.add_module('b2_r3',nn.ReLU(True))
        
        self.bottleneck4 = nn.Sequential()
        self.bottleneck4.add_module('b4_conv1',nn.Conv2d(64, 16, kernel_size=1,stride=1,padding = 'same')) # change with 64 and 16 for digit, 128 and 32 for alpha
        #self.bottleneck4.add_module('b4_r1',nn.ReLU(True))
        self.bottleneck4.add_module('b4_bn1',nn.BatchNorm2d(16))
        self.bottleneck4.add_module('b4_r1',nn.ReLU(True))
        self.bottleneck4.add_module('b4_conv2',nn.Conv2d(16, 16, kernel_size=3,stride=1,padding = 'same', dilation=2))
        #self.bottleneck4.add_module('b4_r2',nn.ReLU(True))
        self.bottleneck4.add_module('b4_bn2',nn.BatchNorm2d(16))
        self.bottleneck4.add_module('b4_r2',nn.ReLU(True))
        self.bottleneck4.add_module('b4_conv3',nn.Conv2d(16, 64, kernel_size=1,stride=1,padding = 'same'))
        #self.bottleneck4.add_module('b4_r3',nn.ReLU(True))
        self.bottleneck4.add_module('b4_bn3',nn.BatchNorm2d(64))
        self.bottleneck4.add_module('b4_r3',nn.ReLU(True))
        #'''
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(256, 100))  #256 for digit 128*4*4 for aplha
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100)) # 500 for alpha
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        #self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10)) #10 for digit and 4 for alpha
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128*4*4, 100))#256 for digit 128*4*4 for aplha
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        #self.conv3 = 
        self.conv4 = nn.Conv2d(384, 128, kernel_size=3,dilation=2) # change back to square filter and dilation 2 for digit
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3,dilation=2)
        self.max3 = nn.MaxPool2d(kernel_size=3,stride=2)
    def forward(self, input_data, alpha):
        #input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.features(input_data)
        #feature = self.feature(feature)
        
        feature = self.relu(self.bn1(self.conv4(feature)))
        feature = self.bottleneck2(feature)
        #feature = feature_o1 + feature_n1
        feature = self.cbam1(feature)
        feature = self.relu(self.bn2(self.conv5(feature)))
        feature = self.bottleneck4(feature)
        #feature = feature_o2 + feature_n2
        feature = self.cbam2(feature)
        feature = self.max3(feature)
        

        feature = feature.view(-1, 256) #256 for digit
        reverse_feature = ReverseLayerF.apply(feature, 1.0)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        #print(class_output.shape)
        return class_output, domain_output
