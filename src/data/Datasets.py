import os
import torch
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from Config import batch_size_val, batch_size_test, LEVIR_dir

# ---------------------- Albumentations数据增强配置 ----------------------

# 验证/测试集变换（仅归一化）
val_test_transform = A.Compose([
    # 添加 Resize 操作（固定尺寸为 256x256）
    # A.Resize(height=256, width=256, interpolation=Image.BILINEAR, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], additional_targets={
    'imageB': 'image',
    'mask': 'mask'
})

# ---------------------- 数据集类定义 ----------------------
class LoveDADataset(Dataset):
    def __init__(self, imageA_dir, imageB_dir, label_dir, transform=None):
        self.imageA_dir = imageA_dir
        self.imageB_dir = imageB_dir
        self.label_dir = label_dir

        # 获取排序后的文件列表
        self.imagesA = sorted(os.listdir(imageA_dir))
        self.imagesB = sorted(os.listdir(imageB_dir))
        self.labels = sorted(os.listdir(label_dir))

        # 验证数据一致性
        assert len(self.imagesA) == len(self.imagesB) == len(self.labels), "各目录文件数量不一致"
        for a, b, l in zip(self.imagesA, self.imagesB, self.labels):
            assert a == b == l, f"文件名不匹配: {a}, {b}, {l}"

        self.transform = transform

    def __len__(self):
        return len(self.imagesA)

    def __getitem__(self, idx):
        img_name = self.imagesA[idx]
        
        # 加载图像
        imageA = np.array(Image.open(os.path.join(self.imageA_dir, img_name)).convert('RGB'))
        imageB = np.array(Image.open(os.path.join(self.imageB_dir, img_name)).convert('RGB'))
        mask = np.array(Image.open(os.path.join(self.label_dir, img_name)).convert('L'))
        
        # 将mask转换为二值标签（假设原始标签0为背景，255为变化区域）
        mask = (mask > 0).astype(np.float32)

        # 应用数据增强
        if self.transform:
            transformed = self.transform(image=imageA, imageB=imageB, mask=mask)
            imageA = transformed['image']
            imageB = transformed['imageB']
            mask = transformed['mask']
        else:
            # 若无变换，直接转换为Tensor
            imageA = torch.from_numpy(imageA).permute(2, 0, 1).float() / 255.0
            imageB = torch.from_numpy(imageB).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        return imageA, imageB, mask, img_name

# ---------------------- 初始化数据集 ----------------------
valid_dataset = LoveDADataset(
    imageA_dir=os.path.join(LEVIR_dir, "val/A"),
    imageB_dir=os.path.join(LEVIR_dir, "val/B"),
    label_dir=os.path.join(LEVIR_dir,  "val/label"),
    transform=val_test_transform
)

test_dataset = LoveDADataset(
    imageA_dir=os.path.join(LEVIR_dir, "test/A"),
    imageB_dir=os.path.join(LEVIR_dir, "test/B"),
    label_dir=os.path.join(LEVIR_dir,  "test/label"),
    transform=val_test_transform
)

# ---------------------- 创建DataLoader ----------------------
valid_loader = DataLoader(valid_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, pin_memory=False)