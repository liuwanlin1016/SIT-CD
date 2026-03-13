import os
import cv2
import shutil
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from src.data.Datasets2  import test_loader# , valid_loader
from src.data.Datasets import test_loader# , valid_loader
from PIL import Image
from tqdm import tqdm
from Config import model_path, document_dir, device, LEVIR_dir
from src.model.EncoderDecoder import model
from torchmetrics import  JaccardIndex, F1Score, Recall, Accuracy, Precision
import numpy as np

# 创建保存目录
value_f1  = F1Score(task="binary", threshold=0.5).to(device)  
value_iou = JaccardIndex(task="binary", threshold=0.5).to(device)
value_recall = Recall(task="binary", threshold=0.5).to(device)
value_acc = Accuracy(task="binary", threshold=0.5).to(device)
value_Precision = Precision(task = "binary", threshold=0.5).to(device)

model = model.to(device)
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_path, map_location=device,weights_only=False),strict=True)

model.eval()
with torch.no_grad():
    for imagesA, imagesB, masks, img_names in tqdm(test_loader, desc=f"验证进度", total=len(test_loader)):
        imagesA = imagesA.to(device)
        imagesB = imagesB.to(device)
        masks  = masks.to(device)
        outputs = model(imagesA,imagesB).to(device) #logits
        outputs = outputs.squeeze(1)

        outputs = outputs.sigmoid() #概率值
        outputs = (outputs > 0.5).float()

        value_f1.update(outputs, masks)
        value_iou.update(outputs, masks)
        value_recall.update(outputs, masks)
        value_acc.update(outputs, masks)  # 新增update
        value_Precision.update(outputs, masks) 


print(
    f' Acc: {value_acc.compute():.6f},'
    f' F1 : {value_f1.compute():.6f},'
    f' IoU: {value_iou.compute():.6f},'
    f' Precision: {value_Precision.compute():.6f},'
    f' Recall: {value_recall.compute():.6f}')

"""
模型输出与基于torchmetric的IoU、F1、Recall指标计算
"""