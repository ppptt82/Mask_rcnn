# Mask R-CNN Project

This project implements Mask R-CNN using PyTorch. It is used for instance segmentation about MoS2 and other chemical crystals.

And I wish it can test on more international and famous datasets, and get a good result.

## Directory Structure

- `config/`: Configuration files for training and testing.
- `data/`: Datasets and annotations.
- `datasets/`: Custom dataset scripts.
- `models/`: Model definitions.
- `utils/`: Utility scripts.

## Setup

- Python environment: Conda
- Deep Learning Framework: PyTorch


## Get Start with Dataset

- COCO format dataset is required, if you have some labelme annotated data, you'd better transform them to COCO format by labelme2coco
- Use "pip install -U labelme2coco".Get more information from https://github.com/fcakyon/labelme2coco
- About num_class, you don't need to annotate the background class, but it still counts one class. It means if you have 5 foreground class, you should input num_class=6.



## Have done

- [09/06] train success! this afternoon i has been tortured by the pytorch and cuda version, it cost me 2 hours.

## ToDO
 
- 训练过程中，记录trainloss和valloss，并实时绘制loss变化图与lr变化图（随epoch变化）
- 训练完毕后，导入指定的模型参数文件，修改model输出，使之在推理时输出bboxes、labels、masks，并设置输入方式（图像文件夹、单张图像等），输出为绘制好的可视化结果，并将bboxes、labels、masks保存至对应的json文件
- 修改lr、batch_size，比较不同的参数对结果的影响
- 数据增强。目前只有100张原始图像，计划增加到500张？（待定）
- loss修改1：添加proco组件
- loss修改2：数据集具有一定的few-shot特征，例如，当模型输出为三角形，但ground_truth中并未给出该实例的标注，通过分析模型输出的mask形状，给予一定的奖励。
