import os
import torch
from datetime import datetime

#检测CUDA
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# batch_size
batch_size_val = 8
batch_size_test  = 8

daily_date = datetime.now().strftime("%m%d%H%M")  # 仅取日期部分

#总文件路径
document_dir = f"./"
# 本地路径配置
LEVIR_dir = "/usr/liu/Datasets/LEVIR-CD"
LEVIR2_dir = "/usr/liu/Datasets/LEVIR2"

# 指定调用位置和权重
model_dir = os.path.join(document_dir, f"checkpoints")
model_name = "model_weights.pth"
model_path = os.path.join(model_dir, model_name)