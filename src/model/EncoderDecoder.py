import sys
from pathlib import Path

# 获取项目根目录（假设文件路径为 ViTUNet_CD/src/model/Model_CD.py）
project_root = Path(__file__).resolve().parent.parent.parent  # 向上三级到根目录
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.InceptionV4_all import InceptionStem, InceptionA, InceptionB, InceptionC, ReductionA, ReductionB
from src.model.Decoder import UpBlock

class Encoder(nn.Module):
    def __init__(self,dropout_rates=0.50):
        super(Encoder, self).__init__()
        self.encoder0 = InceptionStem()

        self.encoder1 = nn.Sequential(
            # 4 Inception-A modules
            # InceptionA(384),
            InceptionA(384),
            # InceptionA(384),
            nn.Dropout2d(0.3),  # 空间 Dropout
            InceptionA(384)
        )
        self.encoder2 = ReductionA(384)

        self.encoder3 = nn.Sequential(
            # 7 Inception-B modules
            InceptionB(1024), 
            # InceptionB(1024), 
            InceptionB(1024), 
            nn.Dropout2d(0.5),  # 空间 Dropout
            # InceptionB(1024), 
            # InceptionB(1024), 
            # InceptionB(1024), 
            InceptionB(1024)
        )
        self.encoder4 = ReductionB(1024)

        self.encoder5 = nn.Sequential(                        
            # 3 Inception-C modules
            InceptionC(1536),
            InceptionC(1536),
            InceptionC(1536),
            nn.Dropout2d(0.7),  # 空间 Dropout
            InceptionC(1536),
            InceptionC(1536)
        )
    def forward(self, x0):
        x1 = self.encoder0(x0)
        x2 = self.encoder1(x1)
        # x2 = x1

        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        # x4 = x3

        x5 = self.encoder4(x4)
        x6 = self.encoder5(x5)
        # x6 = x5

        if __name__ == "__main__":
            print(x0.shape)
            print(x1.shape)
            print(x2.shape)
            print(x3.shape)
            print(x4.shape)
            print(x5.shape)
            print(x6.shape)
            print("stop")
        return x2, x4, x6

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder1 = UpBlock(3072, 2048, 1024,dropout_rates=0.6)
        self.decoder2 = UpBlock(1024, 768, 384,dropout_rates=0.3)
        
        # 第1阶段：上采样2倍 (128×128 -> 256×256)
        self.stage1 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3)
            
        )
        
        # 第2阶段：上采样2倍 (256×256 -> 512×512)
        self.stage2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2)
        )
        
        # 第3阶段：上采样2倍 (512×512 -> 1024×1024)
        self.stage3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        
        # 最终调整通道数 (64 -> 1)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, z6, z4, z2):
        m0 = self.decoder1(z6, z4)
        m1 = self.decoder2(m0, z2)
        m2 = self.stage1(m1)    # [4, 384, 128, 128] -> [4, 256, 256, 256]
        m3 = self.stage2(m2)    # [4, 256, 256, 256] -> [4, 128, 512, 512]
        m4 = self.stage3(m3)    # [4, 128, 512, 512] -> [4, 64, 1024, 1024]
        m5 = self.final_conv(m4) # [4, 64, 1024, 1024] -> [4, 1, 1024, 1024]
        if __name__ == "__main__":
            print(z6.shape)
            print(m0.shape)
            print(m1.shape)
            print(m2.shape)
            print(m3.shape)
            print(m4.shape)
            print(m5.shape)
        
        return m5

class Model_CD(nn.Module):
    def __init__(self):
        super(Model_CD,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input1, input2):
        x2, x4, x6 = self.encoder(input1)
        y2, y4, y6 = self.encoder(input2)

        z6 = torch.cat([x6,y6], dim=1)
        z4 = abs(x4 - y4)
        z2 = abs(x2 - y2)
        output = self.decoder(z6, z4, z2)
        return output

model = Model_CD()

# 示例使用
if __name__ == "__main__":
    # 模拟输入数据
    input1_tensor = torch.randn(4, 3, 1024, 1024) 
    input2_tensor = torch.randn(4, 3, 1024, 1024) 
    
    # 创建模型
    
    
    # 前向传播
    output = model(input1_tensor, input2_tensor)
    
    print("输入形状:", input1_tensor.shape)
    print("输出形状:", output.shape)  # 预期输出: torch.Size([4, 192, 127, 127])
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数数量: {total_params:,}")