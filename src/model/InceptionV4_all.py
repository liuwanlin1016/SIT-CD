import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        return F.relu(x, inplace=True)

class InceptionStem(nn.Module):
    def __init__(self, in_channels=3):
        super(InceptionStem, self).__init__()
        
        # 第一阶段：快速下采样
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 1分支left: 3x3卷积下采样
        self.branch1_right = BasicConv2d(64, 96, kernel_size=3, stride=2, padding=1)
        # 1分支right: maxpool + 1x1卷积
        self.branch1_left = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(64, 64, kernel_size=1,stride=1,padding=0)
        )
        
        # 分支合并后的卷积
        self.conv_cat_1 = BasicConv2d(160, 64, kernel_size=1)
        
        # 2分支left: 1x1->3x3卷积下采样
        self.branch2_left = nn.Sequential(
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        # 2分支right: 1x1->7x1->1x7->3x3卷积下采样
        self.branch2_right = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
        )

        # 3分支left: 3x3卷积下采样
        self.branch3_right = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        # 3分支right: maxpool + 1x1卷积
        self.branch3_left = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BasicConv2d(192, 192, kernel_size=1,stride=1,padding=0)
        )


        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 输入形状: (B, 3, 1024, 1024)
        x = self.conv1(x)  # (B, 32, 512, 512)
        x = self.conv2(x)  # (B, 32, 512, 512)
        x = self.conv3(x)  # (B, 64, 512, 512)
        
        # 并行分支
        branch1_left  = self.branch1_left(x)     # (B, 96, 256, 256)
        branch1_right = self.branch1_right(x)  # (B, 64, 256, 256)
        # 合并分支
        x = torch.cat([branch1_left, branch1_right], dim=1)  # (B, 160, 256, 256)
        x = self.conv_cat_1(x)       # (B, 64, 256, 256)

        # 并行分支
        branch2_left  = self.branch2_left(x)
        branch2_right = self.branch2_right(x)
        # 合并分支
        x = torch.cat([branch2_left, branch2_right], dim=1)  # (B, 192, 256, 256)

        # 并行分支
        branch3_left  = self.branch3_left(x)     # (B, 192, 256, 256)
        branch3_right = self.branch3_right(x)    # (B, 192, 256, 256)
        # 合并分支
        x = torch.cat([branch3_left, branch3_right], dim=1)  # (B, 384, 256, 256)
        
        return x

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
    
        # Branch-1: Avg Pool -> 1x1 conv
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 96, kernel_size=1)
        )

        # Branch-2: 1x1 conv
        self.branch2 = BasicConv2d(in_channels, 96, kernel_size=1)

        # Branch-3: 1x1 conv -> 3x3 conv
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )
        

        # Branch 4: 1x1 conv -> 3x3 conv -> 3x3 conv
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )
            
    def forward(self, x):

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        # Branch-1: Avg Pool -> 1x1 conv
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 128, kernel_size=1)
        )
        # Branch-2: 1x1 conv
        self.branch2 = BasicConv2d(in_channels, 384, kernel_size=1)

        # Branch-3
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )

        #Branch-4
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)

class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        # Branch-1
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_channels, 256, kernel_size=1)
        )

        # Branch-2: 1x1 conv
        self.branch2 = BasicConv2d(in_channels, 256, kernel_size=1)

        # Branch-3
        self.branch3 = BasicConv2d(in_channels, 384, kernel_size=1)

        #Branch-3.1
        self.branch3_1 = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        #Branch-3.2
        self.branch3_2 = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))

        #Branch-4
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )

        #Branch-4.1
        self.branch4_1 = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        #Branch-4.2
        self.branch4_2 = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        
    def forward(self, x):        
        branch1 = self.branch1(x)

        branch2 = self.branch2(x)

        branch3 = self.branch3(x)
        branch3_1 = self.branch3_1(branch3)
        branch3_2 = self.branch3_2(branch3)
        branch3_list = [branch3_1, branch3_2]
        branch3 = torch.cat(branch3_list, dim=1)

        branch4 = self.branch4(x)
        branch4_1 = self.branch4_1(branch4)
        branch4_2 = self.branch4_2(branch4)
        branch4_list = [branch4_1, branch4_2]
        branch4 = torch.cat(branch4_list, dim=1)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)

class ReductionA(nn.Module):
    def __init__(self, in_channels, k=192, l=224, m=256, n=384):
        super(ReductionA, self).__init__()
        self.branch1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.branch2 = BasicConv2d(384, n, kernel_size=3, stride=2, padding=1)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=1),  # 1x1卷积
            BasicConv2d(k, l, kernel_size=3, padding=1),  # 3x3卷积
            BasicConv2d(l, m, kernel_size=3, stride=2, padding=1)  # 3x3卷积步长2
        )
    def forward(self, x):
        # 各分支处理
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        
        # 沿通道维度拼接
        return torch.cat([branch1, branch2, branch3], dim=1)

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1),
            BasicConv2d(192,  192, kernel_size=3, stride=2, padding=1)
        )
            
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),  # 1x1卷积
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        # 各分支处理
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        
        # 沿通道维度拼接
        return torch.cat([branch1, branch2, branch3], dim=1)