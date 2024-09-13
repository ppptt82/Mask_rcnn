import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.ops import box_iou

import time
import os
import logging
from PIL import Image

from datasets.coco_dataset import CustomCocoDataset
from models.maskrcnn_model import get_my_maskrcnn_model
from models.maskrcnn_proco_model import get_my_maskrcnn_proco_model

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# 配置日志管理器
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为DEBUG，这样会打印所有级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置时间格式
    handlers=[
        logging.FileHandler(f"/remote-home/ptt/Mask_rcnn/logs/train_{time.localtime().tm_mon:02d}{time.localtime().tm_mday:02d}{time.localtime().tm_hour}{time.localtime().tm_min}.log"),  # 将日志记录到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)


def get_ap(predictions, targets):
    # 将模型预测的结果和标签转换为COCO格式
    coco_gt = COCO()  # COCO ground truth
    coco_dt = coco_gt.loadRes(predictions)  # COCO detections

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats # stats[0] 是 AP


def train_one_epoch(model, optimizer, train_data_loader, val_data_loader, device, epoch):
    train_loss = 0
    val_loss = 0
    proco_loss = 0
    proco_cost_time = 0
    num_batches = len(train_data_loader)
    start_time = time.time()

    # 训练与更新
    model.train()  # 切换到训练模式
    for i, (images, targets) in enumerate(train_data_loader):
        images = [image.to(device) for image in images]  # 将图像发送到指定设备
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # 将标注发送到指定设备

        loss_dict, detections, _, cost_time  = model(images, targets)  # 返回损失字典
        losses = sum(loss for loss in loss_dict.values()) or torch.tensor(0.0)  # 计算所有损失的总和
        # 反向传播并更新权重
        optimizer.zero_grad()  # 清除之前的梯度
        losses.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数

        train_loss += losses.item()  # 累加损失
        proco_loss += loss_dict["loss_scl"]
        proco_cost_time += cost_time
        

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_data_loader):
            images = [image.to(device) for image in images]  # 将图像发送到指定设备
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # 将标注发送到指定设备

            loss_dict, detections, _, _ = model(images, targets)  # 返回损失字典
            losses = sum(loss for loss in loss_dict.values()) or torch.tensor(0.0) # 计算所有损失的总和

            val_loss += losses.item()

    logging.info( f"Epoch [{epoch + 1}], train_Loss_average: {train_loss/40:.4f}, proco_loss_average: {proco_loss/40:.4f}, val_Loss_average: {val_loss/10:.4f}, epoch_Time: {time.time() - start_time:.2f}s, porco_time: {proco_cost_time:.2f}s")
    # 返回平均损失
    return None


# 主函数
def main():
    # 定义数据路径
    dataset_root = '/remote-home/ptt/Mask_rcnn/data/'
    train_json_path = 'coco_data/train.json'
    val_json_path = 'coco_data/val.json'

    # 定义设备 (GPU 优先)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)

    # 创建自定义数据集
    train_dataset = CustomCocoDataset(root=dataset_root, annotation_file=train_json_path)
    val_dataset = CustomCocoDataset(root=dataset_root, annotation_file=val_json_path)

    # 定义数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    val_data_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

    # 定义模型
    num_classes = 6  # 5 个前景类 + 1 个背景类
    # model = get_my_maskrcnn_model(num_classes)
    model = get_my_maskrcnn_proco_model(num_classes)
    model.load_state_dict(torch.load('/remote-home/ptt/Mask_rcnn/checkpoint/0910_maskrcnn_proco_epoch_181.pth', map_location=device))


    model.to(device)

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 训练模型
    num_epochs = 100
    start_epoch = 181
    for epoch in range(start_epoch, start_epoch + num_epochs):

        train_one_epoch(model, optimizer, train_data_loader, val_data_loader, device, epoch)

        # 保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join("/remote-home/ptt/Mask_rcnn/checkpoint", f"0912_maskrcnn_proco_epoch_{epoch + 1}.pth"))
            logging.info(f"checkpoint mask_rcnn_epoch_{epoch + 1}.pth has saved")
        

if __name__ == "__main__":
    main()
