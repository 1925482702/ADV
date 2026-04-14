# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init_

__all__ = {'Detect', 'Segment', 'Pose', 'Classify', 'OBB', 'RTDETRDecoder', 'ShiftDetect', 'CrossModalShift', 'ShiftHead'}




class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        # self.cv2 = nn.ModuleList(nn.Conv2d(x, 4 * self.reg_max, 1) for x in ch)
        # self.cv3 = nn.ModuleList(nn.Conv2d(x, self.nc, 1) for x in ch)
        #
        # cv2 -> 锚框, cv3 -> 类别
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)

        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if len(x) > 3:
            x = x[3:]

        for i in range(self.nl):
            x[i]= torch.cat((self.cv2[i](x[i]),self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)

        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

        # m = self  # self.model[-1]  # LinearProbe() module
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        #     a.bias.data[:] = 1.0  # box
        #     b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = self.detect(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = self.detect(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes):
        """Decode rotated bounding boxes."""
        return dist2rbox(self.dfl(bboxes), self.angle, self.anchors.unsqueeze(0), dim=1) * self.strides


class ShiftDetect(Detect):
    """
    YOLOv8 Detect head with object-level shift prediction for cross-modal alignment.
    
    每个物体输出：
    - 检测框 (原任务)
    - 类别 (原任务)
    - shift (dx, dy) - 物体在两个模态间的空间偏移
    
    用于强制模型进行跨模态交互，解决模态不平衡问题
    """
    
    def __init__(self, nc=80, ch=()):
        """
        初始化 ShiftDetect
        
        Args:
            nc: 类别数
            ch: 输入通道列表
        """
        super().__init__(nc, ch)
        # Shift 预测分支：每个 anchor 预测 dx, dy
        c_shift = max(16, ch[0] // 4)
        self.cv_shift = nn.ModuleList(
            nn.Sequential(
                Conv(x, c_shift, 3), 
                Conv(c_shift, c_shift, 3), 
                nn.Conv2d(c_shift, 2, 1)  # 输出 dx, dy
            ) for x in ch
        )
        # 输出通道数：训练时 box + cls + shift，推理时 box + cls
        self.no_train = nc + self.reg_max * 4 + 2
        self.no_infer = nc + self.reg_max * 4
        self.no = self.no_train
    
    def forward(self, x):
        """
        前向传播
        
        Returns:
            训练时: (检测特征)
            推理时: ((检测结果, 中间特征))
        """
        # 处理可能的额外输入
        if len(x) > 3:
            x = x[3:]
        
        # 对每个尺度进行预测
        shift_preds = []
        for i in range(self.nl):
            shift_i = self.cv_shift[i](x[i])  # [B, 2, H, W]
            shift_preds.append(shift_i)
            # 拼接: [box_reg, cls, shift]
            x[i] = torch.cat((
                self.cv2[i](x[i]),      # box: [B, reg_max*4, H, W]
                self.cv3[i](x[i]),      # cls: [B, nc, H, W]
                shift_i                 # shift: [B, 2, H, W]
            ), 1)
        
        if self.training:
            # 训练时返回所有尺度的特征（包含 shift）
            return x
        
        # 推理时
        shape = x[0].shape  # BCHW
        # 先分离出 box 和 cls，不使用 shift
        x_cat = []
        for xi in x:
            # xi 的形状是 [B, no_train, H, W]，其中 no_train = nc + reg_max*4 + 2
            # 我们只需要前 no_infer 个通道：box + cls
            xi_no_shift = xi[:, :self.no_infer, :, :]  # [B, nc + reg_max*4, H, W]
            x_cat.append(xi_no_shift.view(shape[0], self.no_infer, -1))
        x_cat = torch.cat(x_cat, 2)  # [B, nc + reg_max*4, total_anchors]
        
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (xx.transpose(0, 1) for xx in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # 分离 box, cls
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)
        
        # 推理时只输出 box 和 cls，不输出 shift
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect.forward

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """
    export = False  # export mode

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hd=256,  # hidden dim
            nq=300,  # num queries
            ndp=4,  # num decoder points
            nh=8,  # num head
            ndl=6,  # num decoder layers
            d_ffn=1024,  # dim of feedforward
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # Training args
            nd=100,  # num denoising
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = \
            get_cdn_group(batch,
                          self.nc,
                          self.num_queries,
                          self.denoising_class_embed.weight,
                          self.num_denoising,
                          self.label_noise_ratio,
                          self.box_noise_scale,
                          self.training)

        embed, refer_bbox, enc_bboxes, enc_scores = \
            self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(embed,
                                              refer_bbox,
                                              feats,
                                              shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0 ** i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float('inf'))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = len(feats)
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init_` would cause NaN when training with custom datasets.
        # linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class CrossModalShift(nn.Module):
    """
    交叉注意力模块：对比两个模态的特征，预测空间偏移
    
    输入:
        - rgb_feat: RGB 特征 [B, C, H, W]
        - ir_feat: IR 特征 [B, C, H, W]
        - shift_modality: [B] tensor, 0=RGB被平移(query=RGB), 1=IR被平移(query=IR)
    
    输出:
        - shift_pred: 每个位置的偏移预测 [B, 2, H, W]
    """
    
    def __init__(self, channels, num_heads=4, downsample=True):
        super().__init__()
        self.num_heads = num_heads
        self.downsample = downsample

        # 可选：降采样以节省显存
        if downsample:
            self.down = nn.AvgPool2d(4, 4)  # 🔥 改为 4x 降采样
            self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        # Q, K, V 投影
        self.q_proj = nn.Conv2d(channels, channels // 2, 1)
        self.k_proj = nn.Conv2d(channels, channels // 2, 1)
        self.v_proj = nn.Conv2d(channels, channels // 2, 1)

        self.out_proj = nn.Conv2d(channels // 2, channels, 1)

        # Shift 预测头
        self.shift_head = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 2, 1)
        )

        # 🔥 可学习温度参数，约束在 [0.5, 2.0] 范围内
        # 0.5 + 1.5 * sigmoid(temperature) ∈ [0.5, 2.0]
        self.temperature = nn.Parameter(torch.ones(1))

        self.scale = (channels // 2 // num_heads) ** -0.5
    
    def forward(self, rgb_feat, ir_feat, shift_modality):
        """
        Args:
            rgb_feat: RGB 特征 [B, C, H, W]
            ir_feat: IR 特征 [B, C, H, W]
            shift_modality: [B] tensor, 0=RGB被平移(query=RGB), 1=IR被平移(query=IR)

        Returns:
            shift_pred: [B, 2, H, W]
        """
        B, C, H, W = rgb_feat.shape

        # 将 shift_modality 移动到与特征图相同的设备上
        shift_modality = shift_modality.to(rgb_feat.device)

        # 根据 shift_modality 确定 query 和 key/value
        # shift_modality=0 → RGB被平移 → query=IR(基准), kv=RGB(被平移)
        # shift_modality=1 → IR被平移 → query=RGB(基准), kv=IR(被平移)
        # 使用未平移的模态作为基准（query），去另一个模态中检索被平移的物体

        query_feat = torch.where(
            shift_modality.view(B, 1, 1, 1).expand(-1, C, H, W) == 0,
            ir_feat,   # RGB 被平移，用 IR 作为 query（基准）
            rgb_feat   # IR 被平移，用 RGB 作为 query（基准）
        )
        kv_feat = torch.where(
            shift_modality.view(B, 1, 1, 1).expand(-1, C, H, W) == 0,
            rgb_feat,  # RGB 被平移，用 RGB 作为 kv（被平移的物体）
            ir_feat   # IR 被平移，用 IR 作为 kv（被平移的物体）
        )
        
        # 可选降采样
        if self.downsample:
            query_feat_down = self.down(query_feat)
            kv_feat_down = self.down(kv_feat)
            _, _, H_d, W_d = query_feat_down.shape
        else:
            query_feat_down = query_feat
            kv_feat_down = kv_feat
            H_d, W_d = H, W
        
        # 投影
        q = self.q_proj(query_feat_down)
        k = self.k_proj(kv_feat_down)
        v = self.v_proj(kv_feat_down)
        
        # 多头注意力
        q = q.view(B, self.num_heads, -1, H_d * W_d).transpose(-1, -2)
        k = k.view(B, self.num_heads, -1, H_d * W_d).transpose(-1, -2)
        v = v.view(B, self.num_heads, -1, H_d * W_d).transpose(-1, -2)

        # 🔥 使用可学习温度参数
        # temp = 0.5 + 1.5 * sigmoid(temperature) ∈ [0.5, 2.0]
        temp = 0.5 + 1.5 * torch.sigmoid(self.temperature)

        # 注意力（使用温度缩放）
        attn = (q @ k.transpose(-1, -2)) * self.scale * temp
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(-1, -2).contiguous().view(B, -1, H_d, W_d)
        
        attended_feat = self.out_proj(out)
        
        # 上采样恢复
        if self.downsample:
            attended_feat = self.up(attended_feat)
        
        # 特征差分
        diff_feat = query_feat - kv_feat
        
        # 拼接预测
        concat_feat = torch.cat([query_feat, attended_feat, diff_feat], dim=1)
        shift_pred = self.shift_head(concat_feat)
        
        return shift_pred


class ShiftHead(nn.Module):
    """
    独立的 Shift 预测头

    接收三个尺度的 RGB 和 IR 独立特征，预测每个位置的跨模态偏移
    使用交叉注意力显式对比两个模态的特征
    """

    def __init__(self, ch=()):
        super().__init__()

        # ch 会是一个包含 6 或 7 个元素的列表
        # - 7 个元素: [obb_ch, rgb_p3, ir_p3, rgb_p4, ir_p4, rgb_p5, ir_p5] (obb_ch 可以忽略)
        # - 6 个元素: [rgb_p3, ir_p3, rgb_p4, ir_p4, rgb_p5, ir_p5]
        if len(ch) == 7:
            # 忽略第一个 OBB 通道数
            ch = ch[1:]
        
        if len(ch) == 6:
            # 提取 RGB 三个尺度的通道数 (比如 256, 512, 1024)
            ch_rgb = (ch[0], ch[2], ch[4])
            ch_ir = (ch[1], ch[3], ch[5])
        else:
            # 兜底默认值
            ch_rgb = (256, 512, 1024)
            ch_ir = (256, 512, 1024)

        # 三个尺度的交叉注意力模块
        self.shift_p3 = CrossModalShift(ch_rgb[0])
        self.shift_p4 = CrossModalShift(ch_rgb[1])
        self.shift_p5 = CrossModalShift(ch_rgb[2])

        self.training = True

        # 添加属性以兼容 v8DetectionLoss 的初始化逻辑
        # 注意：这些属性只是为了欺骗损失函数初始化，实际计算时不会使用
        import torch

        # stride: ShiftHead 连接的是 P3, P4, P5 特征图，对应的下采样倍率是 8, 16, 32
        self.stride = torch.tensor([8., 16., 32.])

        # nc: 类别数（默认为80，从数据集获取）
        self.nc = 80

        # no: 输出通道数（ShiftHead 输出 dx, dy，所以是2）
        self.no = 2

        # reg_max: DFL 通道数（ShiftHead 不使用 DFL，设置为 0）
        self.reg_max = 0

        # nl: 层数量（3个尺度：P3, P4, P5）
        self.nl = 3
    
    def forward(self, x):
        """
        Args:
            x: 如果是列表:
                - [obb_output, rgb_p3, ir_p3, rgb_p4, ir_p4, rgb_p5, ir_p5]
                - obb_output 是 OBB 层的输出 (feats, angle) 或 feats
               或者只有特征:
                - [rgb_p3, ir_p3, rgb_p4, ir_p4, rgb_p5, ir_p5]

        Returns:
            训练时: (obb_output, [shift_p3, shift_p4, shift_p5])
            推理时: obb_output
        """
        # 🔥 检查是否包含 OBB 输出（第一个元素是 tuple 或 OBB 格式）
        obb_output = None
        if isinstance(x, list) and len(x) >= 1:
            first = x[0]
            # OBB 输出格式：(feats, angle) 或 feats (list of tensors)
            if isinstance(first, tuple) or (isinstance(first, list) and len(first) > 0 and 
                                            isinstance(first[0], torch.Tensor) and first[0].dim() == 4):
                obb_output = first
                x = x[1:]  # 剩下的是 shift 特征
        
        # 检查是否有足够的 shift 特征
        if len(x) < 6:
            # 只有 OBB 输出，没有 shift 特征
            return obb_output if obb_output is not None else None

        rgb_p3, ir_p3, rgb_p4, ir_p4, rgb_p5, ir_p5 = x[:6]

        if not self.training:
            # 推理时只返回 OBB 输出
            return obb_output

        # 1. 从自己身上取下刚才挂载的 shift_modality
        modality = getattr(self, 'shift_modality', None)

        # 兜底保护：如果在验证/推理时没有传入，默认全是 0 (RGB被平移)
        if modality is None:
            B = rgb_p3.shape[0]
            import torch
            modality = torch.zeros(B, device=rgb_p3.device, dtype=torch.long)
            # 只在训练时打印警告（排除模型 summary 阶段）
            # if self.training:
            #     print("WARNING: shift_modality not set, using default 0 (RGB shifted)")

        # 2. 把 modality 传给注意力模块
        shift_p3 = self.shift_p3(rgb_p3, ir_p3, modality)  # [B, 2, H, W]
        shift_p4 = self.shift_p4(rgb_p4, ir_p4, modality)
        shift_p5 = self.shift_p5(rgb_p5, ir_p5, modality)

        shift_output = [shift_p3, shift_p4, shift_p5]
        
        # 🔥 返回 (obb_output, shift_output) 元组
        return (obb_output, shift_output) if obb_output is not None else shift_output
