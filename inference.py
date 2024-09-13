import torch
import torchvision.transforms as T
from torchvision.utils import draw_bounding_boxes
from PIL import Image
import matplotlib.pyplot as plt

from models.maskrcnn_model import get_my_maskrcnn_model
from models.maskrcnn_proco_model import get_my_maskrcnn_proco_model
from datasets.coco_dataset import CustomCocoDataset

import time
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 配置日志管理器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为DEBUG，这样会打印所有级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置时间格式
    handlers=[
        logging.FileHandler(f"/remote-home/ptt/Mask_rcnn/logs/inference_{time.localtime().tm_mon:02d}{time.localtime().tm_mday:02d}{time.localtime().tm_hour}{time.localtime().tm_min}.log"),  # 将日志记录到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)


def load_my_maskrcnn_model(checkpoint_path, num_classes, device):
    """
    加载预训练模型和权重。
    """
    model = get_my_maskrcnn_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # 切换到评估模式
    return model

def load_my_maskrcnn_proco_model(checkpoint_path, num_classes, device):
    """
    加载预训练模型和权重。
    """
    model = get_my_maskrcnn_proco_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()  # 切换到评估模式
    return model

def preprocess_image(image_path):
    """
    预处理图像。
    """
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),  # 将图像转换为 Tensor
    ])
    image = transform(image)
    return image.unsqueeze(0)  # 增加 batch 维度

def display_results(image, output, label_colors, label_names):
    """
    使用 OpenCV 绘制结果，并将其保存为图片。
    """
    # 将 Tensor 转换为 NumPy 数组并转为 uint8 类型
    image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # 将 RGB 转换为 BGR 以符合 OpenCV 需求
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 处理并显示边界框、mask等
    for i, (box, label, score, mask) in enumerate(zip(output[0]['boxes'], output[0]['labels'], output[0]['scores'], output[0]['masks'])):
        if score > 0.5:  # 只显示置信度高于阈值的结果
            box = box.cpu().numpy().astype(np.int32)
            mask = mask[0].cpu().numpy()  # mask 的形状是 (1, H, W)，取出维度 1

            # 获取当前 label 的颜色
            color = label_colors.get(label.item(), [255, 255, 255])  # 默认白色

            # 绘制边界框
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color=color, thickness=3)

            # 绘制标签和分数
            text = f'{label_names[label.item()]} {score*100:.0f}%'
            cv2.putText(image, text, (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            # 将 mask 应用于图像，使用颜色进行叠加
            binary_mask = mask > 0.5  # 将 mask 二值化
            colored_mask = np.zeros_like(image, dtype=np.uint8)
            for c in range(3):  # 对 R, G, B 通道应用颜色
                colored_mask[:, :, c] = binary_mask * color[c]

            # 将 mask 叠加到原图像上
            image = cv2.addWeighted(image, 0.9, colored_mask, 0.4, 0)

    # 保存结果图片
    cv2.imwrite("/remote-home/ptt/Mask_rcnn/output/YZJ_5_output (0).png", image)


def main():
    # 配置参数
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"=============use device:{device} for inference=============")

    
    num_classes = 6  # 5 个前景类 + 1 个背景类

    # 加载模型
    checkpoint_path = '/remote-home/ptt/Mask_rcnn/checkpoint/mask_rcnn_epoch_91.pth' 
    model = load_my_maskrcnn_model(checkpoint_path, num_classes, device)
    # checkpoint_path = '/remote-home/ptt/Mask_rcnn/checkpoint/0912_maskrcnn_proco_epoch_281.pth' 
    # model = load_my_maskrcnn_proco_model(checkpoint_path, num_classes, device)
    
    # 预处理图像
    image_path = '/remote-home/ptt/Mask_rcnn/data/MoS2/YZJ_5.png'  # 替换为你自己的图像路径
    image = preprocess_image(image_path)
    
    # 将图像送到设备
    image = image.to(device)
    
    # 推理
    with torch.no_grad():
        _, detections, _ = model(image)
    
        # 定义颜色映射
    label_colors = {
        1: [0, 0, 255],   
        2: [0, 255, 0],    
        3: [255, 0, 0],    
        4: [255, 255, 0],  
        5: [0, 255, 255]
        # BGR
    }
    label_names = {
        1: 'triangle',   
        2: 'concave',    
        3: 'dendrite',    
        4: 'truncated',  
        5: 'hexagon'
        # BGR
    }

    display_results(image.cpu(), detections, label_colors, label_names)

if __name__ == "__main__":
    main()
