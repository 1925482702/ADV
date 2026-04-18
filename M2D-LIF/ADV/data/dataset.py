"""
双模态数据集类
支持RGB + IR图像对的加载和处理
"""

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import Compose, Format, Instances, PairedLetterBox
from .utils import get_hash, img2label_paths, verify_image_label


class BaseDataset(Dataset):
    """
    基础数据集类，支持双模态图像加载
    
    Args:
        img_path (str): RGB图像路径
        imgsz (int): 图像大小
        cache (bool): 是否缓存图像
        augment (bool): 是否应用数据增强
        hyp (dict): 超参数
        prefix (str): 日志前缀
        rect (bool): 是否使用矩形训练
        batch_size (int): 批大小
        stride (int): 步长
        single_cls (bool): 是否单类训练
        classes (list): 包含的类别
        fraction (float): 数据集比例
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=None,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.hyp = hyp if hyp is not None else {}
        
        # 获取图像文件列表（RGB和IR）
        self.im_files, self.ir_files = self.get_img_files(self.img_path)
        
        # 获取标签
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)
        
        self.ni = len(self.labels)
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        
        # 缓冲区
        self.buffer = []
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        
        # 图像缓存
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)
        
        # 构建增强流水线
        self.transforms = self.build_transforms(hyp=self.hyp)

    def get_img_files(self, img_path):
        """
        获取RGB和IR图像文件列表
        
        Returns:
            tuple: (RGB图像列表, IR图像列表)
        """
        try:
            f = []
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
            
            # 过滤图像文件
            img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            im_files = sorted([x for x in f if Path(x).suffix.lower() in img_formats])
            
            assert im_files, f'{self.prefix}No images found in {img_path}'
            
            # 生成IR图像路径（假设IR图像在images_ir目录下）
            ir_files = [x.replace("/images/", "/images_ir/") for x in im_files]
            
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}') from e
        
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
            ir_files = ir_files[:round(len(ir_files) * self.fraction)]
        
        return im_files, ir_files

    def update_labels(self, include_class: Optional[list]):
        """更新标签，只包含指定类别"""
        include_class_array = np.array(include_class).reshape(1, -1) if include_class is not None else None
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """
        加载双模态图像（RGB + IR）
        
        Args:
            i: 图像索引
            rect_mode: 是否使用矩形模式
            
        Returns:
            tuple: (图像, 原始尺寸, 调整后尺寸)
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        
        if im is None:
            if fn.exists():
                try:
                    im = np.load(fn)
                except Exception as e:
                    print(f'{self.prefix}WARNING: Removing corrupt *.npy file {fn} due to: {e}')
                    Path(fn).unlink(missing_ok=True)
                    im = self._load_dual_modality_image(i)
            else:
                im = self._load_dual_modality_image(i)
            
            if im is None:
                raise FileNotFoundError(f'Image Not Found {f}')
            
            h0, w0 = im.shape[:2]
            
            if rect_mode:
                r = self.imgsz / max(h0, w0)
                if r != 1:
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            
            return im, (h0, w0), im.shape[:2]
        
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def _load_dual_modality_image(self, i):
        """
        加载双模态图像并合并
        
        Returns:
            ndarray: 合并后的图像 (H, W, 6)，前3通道为RGB，后3通道为IR
        """
        # 读取RGB图像
        rgb_img = cv2.imread(self.im_files[i])
        if rgb_img is None:
            return None
        
        # 读取IR图像
        ir_img = cv2.imread(self.ir_files[i])
        if ir_img is None:
            # 如果IR图像不存在，复制RGB图像作为IR
            ir_img = rgb_img.copy()
        
        # 合并双模态图像 (RGB在前，IR在后)
        # 注意：cv2.merge会将多个单通道或三通道图像合并
        # 这里我们使用concatenate来合并两个三通道图像
        merged_img = np.concatenate((rgb_img, ir_img), axis=2)
        
        return merged_img

    def cache_images(self, cache):
        """缓存图像到内存或磁盘"""
        b, gb = 0, 1 << 30
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(8) as pool:
            results = pool.imap(fcn, range(self.ni))
            for i, x in enumerate(results):
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                    b += self.ims[i].nbytes

    def cache_images_to_disk(self, i):
        """保存图像为npy文件"""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), self._load_dual_modality_image(i), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """检查内存是否足够缓存图像"""
        import psutil
        b, gb = 0, 1 << 30
        n = min(self.ni, 30)
        for _ in range(n):
            img = self._load_dual_modality_image(random.choice(range(self.ni)))
            if img is None:
                continue
            ratio = self.imgsz / max(img.shape[0], img.shape[1])
            b += img.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available
        return cache

    def __getitem__(self, index):
        """获取数据样本"""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """获取图像和标签"""
        label = deepcopy(self.labels[index])
        label.pop('shape', None)
        
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])
        
        return self.update_labels_info(label)

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)

    def update_labels_info(self, label):
        """更新标签信息，创建Instances对象"""
        bboxes = label.pop('bboxes')
        segments = label.pop('segments', [])
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        
        if segments is None:
            segments = []
        
        if len(segments) > 0:
            segments = np.stack(segments, axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        
        label['instances'] = Instances(bboxes, segments, None, bbox_format=bbox_format, normalized=normalized)
        return label

    def build_transforms(self, hyp=None):
        """构建增强流水线"""
        raise NotImplementedError

    def get_labels(self):
        """获取标签"""
        raise NotImplementedError


class DualModalityDataset(BaseDataset):
    """
    双模态YOLO格式数据集
    
    Args:
        img_path (str): RGB图像路径
        imgsz (int): 图像大小
        cache (bool): 是否缓存
        augment (bool): 是否增强
        hyp (dict): 超参数
        prefix (str): 前缀
        rect (bool): 矩形训练
        batch_size (int): 批大小
        stride (int): 步长
        single_cls (bool): 单类
        classes (list): 类别列表
        fraction (float): 数据集比例
        data (dict): 数据配置字典
    """

    def __init__(self, img_path, imgsz=640, cache=False, augment=True, hyp=None,
                 prefix='', rect=False, batch_size=16, stride=32, single_cls=False,
                 classes=None, fraction=1.0, data=None):
        self.data = data if data is not None else {}
        super().__init__(img_path, imgsz, cache, augment, hyp, prefix, rect,
                        batch_size, stride, single_cls, classes, fraction)

    def get_labels(self):
        """获取YOLO格式标签"""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        
        # 尝试加载缓存
        cache = None
        try:
            if cache_path.exists():
                import gc
                gc.disable()
                cache = np.load(str(cache_path), allow_pickle=True).item()
                gc.enable()
                assert cache['version'] == '1.0.0'
                assert cache['hash'] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache = self.cache_labels(cache_path)
        
        # 检查缓存是否有效
        if cache is None:
            cache = self.cache_labels(cache_path)
        
        # 读取缓存
        nf, nm, ne, nc, n = cache.pop('results')
        print(f'{self.prefix}Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt')
        
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels = cache['labels']
        
        if not labels:
            print(f'WARNING: No images found in {cache_path}')
        
        self.im_files = [lb['im_file'] for lb in labels]
        # 同步更新 ir_files，保持与 im_files 对齐
        self.ir_files = [x.replace("/images/", "/images_ir/") for x in self.im_files]
        return labels

    def cache_labels(self, path):
        """缓存标签"""
        from itertools import repeat
        
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        total = len(self.im_files)
        
        # 创建参数列表
        args = []
        for im_file, lb_file in zip(self.im_files, self.label_files):
            args.append((im_file, lb_file, self.prefix, False, 
                        len(self.data.get('names', [])), 0, 2))
        
        with ThreadPool(8) as pool:
            results = pool.starmap(verify_image_label, args)
            
            for i, result in enumerate(results):
                im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = result
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file and lb is not None:
                    x['labels'].append({
                        'im_file': im_file,
                        'shape': shape,
                        'cls': lb[:, 0:1],
                        'bboxes': lb[:, 1:],
                        'segments': segments,
                        'keypoints': keypoint,
                        'normalized': True,
                        'bbox_format': 'xywh'
                    })
                if msg:
                    msgs.append(msg)
                if (i + 1) % 100 == 0:
                    print(f'{self.prefix}Scanning {i + 1}/{total} images...')
        
        if msgs:
            print('\n'.join(msgs))
        
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs
        x['version'] = '1.0.0'
        
        # 保存缓存
        np.save(str(path), x)
        if path.with_suffix('.cache.npy').exists():
            path.with_suffix('.cache.npy').rename(path)
        print(f'{self.prefix}New cache created: {path}')
        
        return x

    def build_transforms(self, hyp=None):
        """构建双模态增强流水线"""
        from .augment import build_paired_transforms
        
        if hyp is None:
            hyp = {}
        
        if self.augment:
            hyp.setdefault('mosaic', 1.0)  # Mosaic 增强
            hyp.setdefault('mixup', 0.0)
            hyp.setdefault('copy_paste', 0.0)
            hyp.setdefault('degrees', 0.0)
            hyp.setdefault('translate', 0.1)
            hyp.setdefault('scale', 0.5)
            hyp.setdefault('shear', 0.0)
            hyp.setdefault('perspective', 0.0)
            hyp.setdefault('flipud', 0.0)
            hyp.setdefault('fliplr', 0.5)
            hyp.setdefault('hsv_h', 0.015)
            hyp.setdefault('hsv_s', 0.7)
            hyp.setdefault('hsv_v', 0.4)
            
            if self.rect:
                hyp['mosaic'] = 0.0
                hyp['mixup'] = 0.0
            
            transforms = build_paired_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([PairedLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        
        transforms.append(
            Format(bbox_format='xywh', normalize=True, return_mask=False,
                   return_keypoint=False, return_obb=False, batch_idx=True)
        )
        
        return transforms

    def close_mosaic(self, hyp):
        """关闭mosaic增强"""
        hyp['mosaic'] = 0.0
        hyp['copy_paste'] = 0.0
        hyp['mixup'] = 0.0
        self.transforms = self.build_transforms(hyp)

    @staticmethod
    def collate_fn(batch):
        """批次整理函数"""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls', 'segments', 'obb']:
                # 🚨 关键修正：过滤掉None值和空列表
                filtered_value = [v for v in value if v is not None and (isinstance(v, torch.Tensor) and v.numel() > 0)]
                if filtered_value:
                    value = torch.cat(filtered_value, 0)
                else:
                    # 如果所有值都是None或空，创建一个空的tensor
                    value = torch.zeros((0, 0))
            new_batch[k] = value

        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i
        if new_batch['batch_idx']:
            new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        else:
            new_batch['batch_idx'] = torch.zeros((0,), dtype=torch.long)

        return new_batch