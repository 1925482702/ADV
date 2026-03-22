"""
双模态数据增强类
支持RGB + IR图像对的同步增强
"""

import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch


# ==================== 基础类 ====================

class Instances:
    """
    实例类，用于存储边界框、分割等信息
    """
    
    def __init__(self, bboxes, segments, keypoints=None, bbox_format='xywh', normalized=True):
        self.bboxes = np.asarray(bboxes, dtype=np.float32)
        self.segments = np.asarray(segments, dtype=np.float32)
        self.keypoints = keypoints
        self.normalized = normalized
        
        if self.bboxes.ndim == 1:
            self.bboxes = self.bboxes[None, :]
        
        self._bbox_format = bbox_format

    def convert_bbox(self, format='xyxy'):
        """转换边界框格式"""
        if format == self._bbox_format:
            return
        
        if self._bbox_format == 'xywh' and format == 'xyxy':
            self.bboxes[:, 0] -= self.bboxes[:, 2] / 2
            self.bboxes[:, 1] -= self.bboxes[:, 3] / 2
            self.bboxes[:, 2] += self.bboxes[:, 0]
            self.bboxes[:, 3] += self.bboxes[:, 1]
        elif self._bbox_format == 'xyxy' and format == 'xywh':
            self.bboxes[:, 2] -= self.bboxes[:, 0]
            self.bboxes[:, 3] -= self.bboxes[:, 1]
            self.bboxes[:, 0] += self.bboxes[:, 2] / 2
            self.bboxes[:, 1] += self.bboxes[:, 3] / 2
        
        self._bbox_format = format

    def denormalize(self, w, h):
        """反归一化边界框"""
        if not self.normalized:
            return
        
        if self._bbox_format == 'xywh':
            self.bboxes[:, [0, 2]] *= w
            self.bboxes[:, [1, 3]] *= h
        elif self._bbox_format == 'xyxy':
            self.bboxes[:, [0, 2]] *= w
            self.bboxes[:, [1, 3]] *= h
        
        self.normalized = False

    def normalize(self, w, h):
        """归一化边界框"""
        if self.normalized:
            return
        
        if self._bbox_format == 'xywh':
            self.bboxes[:, [0, 2]] /= w
            self.bboxes[:, [1, 3]] /= h
        elif self._bbox_format == 'xyxy':
            self.bboxes[:, [0, 2]] /= w
            self.bboxes[:, [1, 3]] /= h
        
        self.normalized = True

    def scale(self, scale_w, scale_h):
        """缩放边界框"""
        if self._bbox_format == 'xywh':
            self.bboxes[:, [0, 2]] *= scale_w
            self.bboxes[:, [1, 3]] *= scale_h
        elif self._bbox_format == 'xyxy':
            self.bboxes[:, [0, 2]] *= scale_w
            self.bboxes[:, [1, 3]] *= scale_h

    def add_padding(self, padw, padh):
        """添加填充"""
        if self._bbox_format == 'xywh':
            self.bboxes[:, 0] += padw
            self.bboxes[:, 1] += padh
        elif self._bbox_format == 'xyxy':
            self.bboxes[:, [0, 2]] += padw
            self.bboxes[:, [1, 3]] += padh

    def update(self, bboxes=None):
        """更新边界框"""
        if bboxes is not None:
            self.bboxes = np.asarray(bboxes, dtype=np.float32)
            if self.bboxes.ndim == 1:
                self.bboxes = self.bboxes[None, :]

    def __len__(self):
        return len(self.bboxes)
    
    @staticmethod
    def concatenate(instances_list, axis=0):
        """静态方法：连接多个Instances"""
        if not instances_list:
            return instances_list[0]
        
        bboxes = np.concatenate([inst.bboxes for inst in instances_list], axis=axis)
        segments = np.concatenate([inst.segments for inst in instances_list], axis=axis)
        
        result = Instances(bboxes, segments, None, bbox_format='xyxy', normalized=False)
        return result

    def clip(self, imgsz_h, imgsz_w):
        """裁剪边界框到图像范围内"""
        if self._bbox_format == 'xyxy':
            np.clip(self.bboxes[:, 0], 0, imgsz_w, out=self.bboxes[:, 0])
            np.clip(self.bboxes[:, 1], 0, imgsz_h, out=self.bboxes[:, 1])
            np.clip(self.bboxes[:, 2], 0, imgsz_w, out=self.bboxes[:, 2])
            np.clip(self.bboxes[:, 3], 0, imgsz_h, out=self.bboxes[:, 3])

    def remove_zero_area_boxes(self):
        """移除面积为0的边界框"""
        if self._bbox_format == 'xyxy':
            keep = (self.bboxes[:, 2] > self.bboxes[:, 0]) & (self.bboxes[:, 3] > self.bboxes[:, 1])
        elif self._bbox_format == 'xywh':
            keep = (self.bboxes[:, 2] > 0) & (self.bboxes[:, 3] > 0)
        
        self.bboxes = self.bboxes[keep]
        self.segments = self.segments[keep] if len(self.segments) > 0 else self.segments
        
        return keep


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms):
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
    def append(self, transform):
        """添加变换"""
        self.transforms.append(transform)


# ==================== 双模态LetterBox ====================

class PairedLetterBox:
    """
    双模态LetterBox - 同时对RGB和IR图像进行调整和填充
    """
    
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center

    def __call__(self, labels=None, image=None):
        """应用LetterBox变换"""
        if labels is None:
            labels = {}
        
        img = labels.get("img") if image is None else image
        
        # 分离RGB和IR
        img_rgb = img[:, :, :3]
        img_ir = img[:, :, 3:]
        
        shape = img_rgb.shape[:2]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)
        
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        
        if self.center:
            dw /= 2
            dh /= 2
        
        # 调整大小
        if shape[::-1] != new_unpad:
            img_rgb = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
            img_ir = cv2.resize(img_ir, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        
        # 添加填充
        img_rgb = cv2.copyMakeBorder(img_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        img_ir = cv2.copyMakeBorder(img_ir, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # 更新标签
        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = np.concatenate((img_rgb, img_ir), axis=2)
            labels["resized_shape"] = new_shape
            return labels
        else:
            return np.concatenate((img_rgb, img_ir), axis=2)

    def _update_labels(self, labels, ratio, padw, padh):
        """更新标签坐标"""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


# ==================== 双模态Mosaic ====================

class PairedMosaic:
    """
    双模态Mosaic增强 - 将4张图像拼成一张
    """
    
    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        self.dataset = dataset
        self.imgsz = imgsz
        self.p = p
        self.n = n
        self.border = (-imgsz // 2, -imgsz // 2)

    def __call__(self, labels):
        """应用Mosaic增强"""
        if random.uniform(0, 1) > self.p:
            return labels
        
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]
        
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        labels["mix_labels"] = mix_labels
        
        if self.n == 4:
            return self._mosaic4(labels)
        else:
            return labels

    def get_indexes(self):
        """获取随机索引"""
        return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mosaic4(self, labels):
        """创建2x2 Mosaic"""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            
            # 分离RGB和IR
            img_rgb = img[:, :, :3]
            img_ir = img[:, :, 3:]
            
            h, w = labels_patch.pop("resized_shape")
            
            if i == 0:
                img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
                img4_ir = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]
            img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]
            
            padw = x1a - x1b
            padh = y1a - y1b
            
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = np.concatenate((img4, img4_ir), axis=2)
        return final_labels

    def _update_labels(self, labels, padw, padh):
        """更新标签"""
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """合并标签"""
        if len(mosaic_labels) == 0:
            return {}
        
        cls = []
        instances = []
        imgsz = self.imgsz * 2
        
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        
        return final_labels


# ==================== 双模态MixUp ====================

class PairedMixUp:
    """
    双模态MixUp增强 - 混合两张图像
    """
    
    def __init__(self, dataset, pre_transform=None, p=0.0):
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """应用MixUp增强"""
        if random.uniform(0, 1) > self.p:
            return labels
        
        index = random.randint(0, len(self.dataset) - 1)
        labels2 = self.dataset.get_image_and_label(index)
        
        r = np.random.beta(32.0, 32.0)
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        
        return labels


# ==================== 双模态Albumentations ====================

class PairedAlbumentations:
    """
    双模态Albumentations增强 - 同时对RGB和IR应用相同的变换
    """
    
    def __init__(self, p=1.0):
        self.p = p
        self.transform = None

    def __call__(self, labels):
        """应用Albumentations增强"""
        img = labels["img"]
        img_rgb = img[:, :, :3]
        img_ir = img[:, :, 3:]
        
        cls = labels["cls"]
        
        if len(cls) > 0 and self.transform and random.random() < self.p:
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*img_rgb.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            
            # 对RGB和IR应用相同的变换
            new = self.transform(image=img_rgb, bboxes=bboxes, class_labels=cls)
            new_ir = self.transform(image=img_ir, bboxes=bboxes, class_labels=cls)
            
            if len(new["class_labels"]) > 0:
                labels["img"] = np.concatenate((new["image"], new_ir["image"]), axis=2)
                labels["cls"] = np.array(new["class_labels"])
                bboxes = np.array(new["bboxes"], dtype=np.float32)
            
            labels["instances"].update(bboxes=bboxes)
        
        return labels


# ==================== 双模态随机翻转 ====================

class PairedRandomFlip:
    """
    双模态随机翻转 - 同时翻转RGB和IR
    """
    
    def __init__(self, direction='horizontal', p=0.5, flip_idx=None):
        self.direction = direction
        self.p = p
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """应用随机翻转"""
        if random.uniform(0, 1) > self.p:
            return labels
        
        img = labels["img"]
        img_rgb = img[:, :, :3]
        img_ir = img[:, :, 3:]
        
        if self.direction == 'horizontal':
            img_rgb = np.fliplr(img_rgb)
            img_ir = np.fliplr(img_ir)
            w = img_rgb.shape[1]
            self._flip_bbox(labels["instances"], w)
        elif self.direction == 'vertical':
            img_rgb = np.flipud(img_rgb)
            img_ir = np.flipud(img_ir)
            h = img_rgb.shape[0]
            self._flip_bbox(labels["instances"], h, vertical=True)
        
        labels["img"] = np.concatenate((img_rgb, img_ir), axis=2)
        return labels

    def _flip_bbox(self, instances, size, vertical=False):
        """翻转边界框"""
        if vertical:
            instances.bboxes[:, 1] = size - instances.bboxes[:, 1]
            instances.segments[:, :, 1] = size - instances.segments[:, :, 1]
        else:
            instances.bboxes[:, 0] = size - instances.bboxes[:, 0]
            instances.segments[:, :, 0] = size - instances.segments[:, :, 0]


# ==================== 双模态随机透视 ====================

class PairedRandomPerspective:
    """
    双模态随机透视变换 - 同时对RGB和IR应用相同的透视变换
    """
    
    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, 
                 perspective=0.0, border=(0, 0), pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        self.pre_transform = pre_transform

    def __call__(self, labels):
        """应用随机透视变换"""
        if self.pre_transform and random.uniform(0, 1) > 0.5:
            return labels
        
        img = labels["img"]
        img_rgb = img[:, :, :3]
        img_ir = img[:, :, 3:]
        
        # 应用相同的透视变换
        M, s = self._get_transform_matrix(img_rgb.shape)
        
        img_rgb = cv2.warpPerspective(img_rgb, M, dsize=(img_rgb.shape[1], img_rgb.shape[0]))
        img_ir = cv2.warpPerspective(img_ir, M, dsize=(img_ir.shape[1], img_ir.shape[0]))
        
        # 更新边界框
        self._update_bbox(labels["instances"], M, img_rgb.shape)
        
        labels["img"] = np.concatenate((img_rgb, img_ir), axis=2)
        return labels

    def _get_transform_matrix(self, shape):
        """获取变换矩阵"""
        # 简化版本，只实现基本的变换
        h, w = shape[:2]
        center = np.array([w / 2, h / 2], dtype=np.float32)
        
        # 缩放
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        scale_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
        
        # 旋转
        angle = random.uniform(-self.degrees, self.degrees)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 平移
        tx = random.uniform(-self.translate, self.translate) * w
        ty = random.uniform(-self.translate, self.translate) * h
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 组合变换
        M = translation_matrix @ rotation_matrix @ scale_matrix
        
        return M, scale

    def _update_bbox(self, instances, M, shape):
        """更新边界框"""
        if len(instances.bboxes) == 0:
            return
        
        # 转换为绝对坐标
        instances.convert_bbox('xyxy')
        if instances.normalized:
            instances.denormalize(shape[1], shape[0])
        
        # 应用透视变换
        corners = instances.bboxes.reshape(-1, 2)
        corners = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
        transformed = corners @ M.T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        
        instances.bboxes = transformed.reshape(-1, 4)


# ==================== 双模态CopyPaste ====================

class PairedCopyPaste:
    """
    双模态CopyPaste增强 - 从其他图像复制实例
    """
    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, labels):
        """应用CopyPaste增强"""
        if random.uniform(0, 1) > self.p:
            return labels
        
        # 简化版本，暂不实现
        return labels


# ==================== Format类 ====================

class Format:
    """
    格式化输出 - 将数据转换为模型输入格式
    """
    
    def __init__(self, bbox_format='xywh', normalize=True, return_mask=False,
                 return_keypoint=False, return_obb=False, batch_idx=True):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.batch_idx = batch_idx

    def __call__(self, labels):
        """格式化数据"""
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)
        
        # 转换图像格式
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)

        # 🚨 关键修正：将图像转换为float32类型并归一化到[0,1]范围
        # YOLOv8期望输入是float32类型，而不是uint8
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        labels["img"] = img
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        
        return labels


# ==================== 构建双模态增强流水线 ====================

def build_paired_transforms(dataset, imgsz, hyp):
    """
    构建双模态增强流水线
    
    Args:
        dataset: 数据集对象
        imgsz: 图像大小
        hyp: 超参数字典
        
    Returns:
        Compose对象
    """
    pre_transform = Compose([
        PairedMosaic(dataset, imgsz=imgsz, p=hyp.get('mosaic', 1.0)),
        PairedCopyPaste(p=hyp.get('copy_paste', 0.0)),
        PairedRandomPerspective(
            degrees=hyp.get('degrees', 0.0),
            translate=hyp.get('translate', 0.1),
            scale=hyp.get('scale', 0.5),
            shear=hyp.get('shear', 0.0),
            perspective=hyp.get('perspective', 0.0),
            pre_transform=None,
        ),
    ])
    
    return Compose([
        pre_transform,
        PairedMixUp(dataset, pre_transform=pre_transform, p=hyp.get('mixup', 0.0)),
        PairedAlbumentations(p=1.0),
        PairedLetterBox(new_shape=(imgsz, imgsz)),  # 🚨 添加 LetterBox 确保输出正方形
        PairedRandomFlip(direction="vertical", p=hyp.get('flipud', 0.0)),
        PairedRandomFlip(direction="horizontal", p=hyp.get('fliplr', 0.5)),
    ])