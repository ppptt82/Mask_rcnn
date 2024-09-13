import torch
from torch import nn, Tensor
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import warnings
import logging

# add for box_features
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import *
from torchvision.ops import MultiScaleRoIAlign

from models.loss.proco import ProCoLoss
from models.loss.logitadjust import LogitAdjust

import time


# 重写self.roi_heads.postprocess_detextions方法，使其输出匹配的特征值的index，以便从1024*1024的box_features中提取出有用的特征向量
def My_roi_heads_postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)  # regression为调整量，调整proposals区域得到最终bbox

        pred_scores = F.softmax(class_logits, -1)  # shape=[1024, 6]

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)  # shape=((512, 6), (512, 6))
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label; 移除预测结果中背景类的预测值
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels


def My_select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets


def My_roi_heads_forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            # gt_proposals_num = [t["boxes"].shape[0] for t in targets]  # 记录了gt中bbox的个数，proposals中后num个区域就取自gt
            # proposals_shape = [proposals[0].shape[0], proposals[1].shape[0]]
        else:
            if targets == None:
                labels = None
                regression_targets = None
                matched_idxs = None
            else:
                proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)


        # 每张图像生成512个proposals
        # features['0'].shape = (2,256,128,168),256为通道数
        # 返回box_features形状为(roi数, 通道数, 特征图尺寸), 此处为（1024, 256, 7, 7)
        raw_box_features_77 = self.box_roi_pool(features, proposals, image_shapes) 

        # 此处返回box_features形状为1024*1024, 由faster_rcnn.py中representation_size = 1024定义
        box_features = self.box_head(raw_box_features_77)

            # 此处应用proco组件
            # proco_features = torch.cat([box_features[:proposals_shape[0]][-gt_proposals_num[0]:], box_features[proposals_shape[0]:][-gt_proposals_num[1]:]]).cuda()
            # proco_targets = torch.cat([targets[0]['labels'], targets[1]['labels']]).cuda()
        start_time = time.time()

        if self.training:
            proco_features = box_features
            proco_targets = torch.cat(labels)

            contrast_logits = self.criterion_scl(proco_features, proco_targets)
            scl_loss = self.criterion_ce(contrast_logits, proco_targets)

        proco_cost_time = time.time() - start_time


        # class_logits形状为1024*6, box_regression形状为1024*24, 即记录了2个图像，每个图像的512个proposals分别属于什么类以及bbox位置
        class_logits, box_regression = self.box_predictor(box_features)
        class_logits.view(-1)

        result: List[Dict[str, torch.Tensor]] = []
        result_of_training: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training :
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

            class_logits = class_logits + contrast_logits * 0.01

            # 下列输出为模型预测结果的输出，在原代码中trian模式并不会输出预测结果
            boxes_of_training, scores_of_training, labels_of_training = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images_of_training = len(boxes_of_training)
            for i in range(num_images_of_training):
                result_of_training.append(
                    {
                        "boxes": boxes_of_training[i],
                        "labels": labels_of_training[i],
                        "scores": scores_of_training[i],
                    }
                )
        else:
            if targets is not None:
                if labels is None:
                    raise ValueError("labels cannot be None")
                if regression_targets is None:
                    raise ValueError("regression_targets cannot be None")
                loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training or (targets is not None):
                if matched_idxs is None:
                    raise ValueError("if in training, matched_idxs should not be None")

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                mask_labels = []
                for img_id in range(num_images):
                    if self.training:
                        pos = torch.where(labels[img_id] > 0)[0]
                        mask_proposals.append(proposals[img_id][pos])
                        pos_matched_idxs.append(matched_idxs[img_id][pos])
                        mask_labels.append(labels[img_id][pos])
                    else:
                        pos = torch.where(labels[img_id] > 0)[0]
                        mask_proposals.append(proposals[img_id][pos])
                        pos_matched_idxs.append(matched_idxs[img_id][pos])
                        mask_labels.append(labels[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training or (targets is not None):
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    raise ValueError("targets, pos_matched_idxs, mask_logits cannot be None when training")

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = {"loss_mask": rcnn_loss_mask}

                masks_probs = maskrcnn_inference(mask_logits, mask_labels)
                for mask_prob, r in zip(masks_probs, result_of_training):
                    r["masks"] = mask_prob
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        # 一般不会进入这个if
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                if matched_idxs is None:
                    raise ValueError("if in trainning, matched_idxs should not be None")

                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                if targets is None or pos_matched_idxs is None:
                    raise ValueError("both targets and pos_matched_idxs should not be None when in training mode")

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                if keypoint_logits is None or keypoint_proposals is None:
                    raise ValueError(
                        "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                    )

                keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps
            losses.update(loss_keypoint)


        if self.training:
            losses["loss_scl"] = scl_loss
            return result_of_training, losses, raw_box_features_77, proco_cost_time
        else:
            return result, losses, raw_box_features_77, proco_cost_time


# 自定义 MaskRCNN 模型
class MaskRCNNWithProCo(MaskRCNN):
    def __init__(self, backbone, num_classes, min_size:int =800, max_size:int =1333):
        super().__init__(backbone=backbone, num_classes=num_classes, 
                         min_size=min_size, max_size=max_size)

        self.roi_heads.forward = My_roi_heads_forward.__get__(self.roi_heads)
        self.roi_heads.postprocess_detections = My_roi_heads_postprocess_detections.__get__(self.roi_heads)
        self.roi_heads.criterion_scl = ProCoLoss(contrast_dim=1024, temperature=0.07, num_classes=6).cuda()
        self.roi_heads.criterion_ce = LogitAdjust([1309, 160, 119, 362, 52]).cuda()
        self.roi_heads.select_training_samples = My_select_training_samples.__get__(self.roi_heads)

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                # logging.info("You have rewriten the forward function so that you don't need targets when training.")
                logging.info("You have rewriten the forward function but you still need targets when training.")
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        
        # get the H,W of images
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]

            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # 将输入图像缩放至min_size,max_size
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # feature_map
        # features['0'].shape=(2,256,128,168),
        # features['1'].shape=(2,256,64,84),
        # ...2, 3
        # features['pool'].shape=(2,256,8,11)
        features = self.backbone(images.tensors)  

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # proposals' shape is (batch_size*num_proposals*4), 4 means bbox points
        # sometimes rpn won't give u proposals, it is just null
        proposals, proposal_losses = self.rpn(images, features, targets)



        # from torchvision.models.detection import RoIHeads
        # self.roi_heads = RoIHeads(para comes from faster_rcnn.py)
        detections, detector_losses, box_features_77, proco_cost_time = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # if torch.jit.is_scripting():
        #     if not self._has_warned:
        #         warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
        #         self._has_warned = True
        #     return losses, detections
        # else:
        #     return self.eager_outputs(losses, detections)

        if self.training:
            return losses, detections, box_features_77, proco_cost_time
        else:
            return losses, detections, box_features_77, proco_cost_time



def get_my_maskrcnn_proco_model(num_classes):
    # 加载预训练的 Mask R-CNN 模型

    # add box features
    # Backbone 设置
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)

    model = MaskRCNNWithProCo(backbone=backbone, num_classes=num_classes, min_size=486, max_size=648)  

    # old version
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Mask R-CNN 默认适用于 COCO 数据集（91 个类）。根据自己的数据集修改分类器和掩码预测器，使其适应新的类别数量.如果数据集有 1 个前景类 + 1 个背景类，num_classes 应设置为 2。
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # 获取掩码分割器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # 替换掩码预测器，适应我们的数据集
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


if __name__=="__main__":
    num_classes = 6  # 5 个前景类 + 1 个背景类
    model = get_my_maskrcnn_proco_model(num_classes)