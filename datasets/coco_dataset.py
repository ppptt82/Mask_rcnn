import os
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import cv2

class CustomCocoDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        """
        自定义 COCO 数据集类。
        :param root: 包含图像文件的目录
        :param annotation_file: COCO 格式的标注文件(JSON 文件)
        :param transforms: 用于数据增强或预处理的可选变换
        """
        self.root = root
        self.coco = COCO(os.path.join(root, annotation_file))  # 加载 COCO 格式的标注文件
        self.ids = list(self.coco.imgs.keys())  # 获取所有图像的 ID
        self.transforms = transforms
        self.to_tensor = ToTensor()


    def __len__(self):
        """
        返回数据集中的图像数量。
        """
        return len(self.ids)


    def __getitem__(self, index):
        """
        根据索引获取图像及其对应的标注信息。
        :param index: 图像的索引
        :return: 图像和标注信息
        """
        # 获取图像 ID
        img_id = self.ids[index]
        
        # 获取图像信息
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        
        # 打开图像
        img = Image.open(img_path).convert("RGB")
        img = self.to_tensor(img) 
        
        # 获取该图像对应的标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # 提取边界框、类别标签和分割掩码
        boxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            # 提取边界框信息，COCO 格式 bbox 为 [x_min, y_min, width, height]
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            
            # 提取类别 ID
            labels.append(ann['category_id'])
            
            # 提取分割掩码并转换为像素掩码
            if 'segmentation' in ann:
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                
                for segmentation in ann['segmentation']:
                    # 转换多边形为二维整数数组
                    poly = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
                    # 使用 OpenCV 绘制多边形掩码
                    cv2.fillPoly(mask, [poly], color=1)
                
                # 将该目标的掩码添加到列表
                masks.append(mask)
        
        # 将边界框、标签转换为 PyTorch 张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}
        
        # 如果有分割掩码，则处理并添加到 target 中
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        target["masks"] = masks
        
        # 应用图像变换（如果有）
        if self.transforms:
            img, target = self.transforms(img, target)
        
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    root = '/remote-home/ptt/Mask_rcnn/data'  # 图像文件夹路径
    annotation_file = 'coco_data/val.json'  # 标注文件路径

    # 初始化自定义数据集
    dataset = CustomCocoDataset(root=root, annotation_file=annotation_file)

    # img, target = dataset.__getitem__(1)

    # print(img)
    # print(target)
    # 使用 PyTorch DataLoader 加载数据
    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=collate_fn)

    # 获取数据
    for images, targets in data_loader:
        print(images)
        print(targets)
        break  # 只打印一个 batch
