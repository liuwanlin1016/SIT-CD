import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class InitModule(nn.Module):
    """ 基础模块，包含通用权重初始化方法 """
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class DoubleAttention(InitModule):
    """双注意力机制（通道+空间）"""
    def __init__(self, in_ch, ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, max(4, in_ch//ratio), 1),
            nn.GELU(),
            nn.Conv2d(max(4, in_ch//ratio), in_ch, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_att(x)
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        
        return x * channel_att * spatial_att

class UpBlock(InitModule): # Green + Blue 
    def __init__(self, in_ch, mid_ch, out_ch, ratio=16, dropout_rates=0.35):
        super().__init__()
        self.piccut = nn.Sequential(
            nn.ConvTranspose2d(in_ch , out_ch, kernel_size = 2, stride = 2,padding=0),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),  # 1x1 卷积调整通道数
            nn.Dropout2d(dropout_rates)  # 空间 Dropout
        )
        self.tconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),  # 平滑特征
            nn.Dropout2d(dropout_rates),  # 空间 Dropout ###
            nn.ReLU()###
        )

        # SE注意力模块
        self.attn = DoubleAttention(in_ch=mid_ch)

        self.dconv  = nn.Sequential(
            nn.Conv2d(mid_ch,out_ch,kernel_size=3,stride=1,padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),  # 1x1 卷积调整通道数
            nn.Dropout2d(dropout_rates),  # 空间 Dropout
            nn.ReLU()###
        )
        # 初始化权重
        self._initialize_weights()

    def forward(self, x, res):
        x_cut = self.piccut(x)
        x     = self.tconv(x)        
        com = torch.cat((res , x), dim = 1)
        attn = self.attn(com)

        final = self.dconv(attn)
        return final + x_cut*0.5
