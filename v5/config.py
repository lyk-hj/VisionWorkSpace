from easydict import EasyDict as edict
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = edict()


# for ClsModel
cfg.Cls = edict()
# for MTClsModel
cfg.MTCls = edict()
# for BackBone
cfg.BaBo = edict()
# for RPN
cfg.RPN = edict()
# for Head
cfg.Head = edict()
# for generate RGB image
cfg.generate = edict()
# for generate fruit image
cfg.fgenerate = edict()
# for anchor generate
cfg.anchor = edict()

cfg.Cls.c0_inc = 3
cfg.Cls.c1_inc = 8
cfg.Cls.c1_ouc = 16
cfg.Cls.c2_ouc = 32
cfg.Cls.c3_ouc = 64
cfg.Cls.c4_ouc = 64
cfg.Cls.c5_inc = 256
cfg.Cls.c5_ouc = 32
cfg.Cls.classes = 9
cfg.Cls.fclasses = 24

cfg.MTCls.c1_inc = 1
cfg.MTCls.c1_ouc = 16
cfg.MTCls.c2_ouc = 32
cfg.MTCls.c3_ouc = 64
cfg.MTCls.c4_ouc = 64
cfg.MTCls.c5_inc = 256
cfg.MTCls.c5_ouc = 32
cfg.MTCls.classes = 8

cfg.BaBo.c0_inc = 3
cfg.BaBo.c1_inc = 4
cfg.BaBo.c1_ouc = 16
cfg.BaBo.c2_ouc = 32
cfg.BaBo.c3_ouc = 64
cfg.BaBo.c4_ouc = 128
cfg.BaBo.i1_ouc = 128
cfg.BaBo.i2_ouc = 512

cfg.RPN.batch_size = 1  # image_counts for training
cfg.RPN.c1_ouc = 512
cfg.RPN.anchors = 9
cfg.RPN.cls_ouc = 2*cfg.RPN.anchors
cfg.RPN.reg_ouc = 4*cfg.RPN.anchors

cfg.Head.batch_size = 256
# cfg.Head.c1_inc = cfg.BaBo.i2_ouc
# cfg.Head.c1_ouc = 1024
# cfg.Head.c2_ouc = 256
cfg.Head.spp = [5]
# cfg.Head.fc1_ins = np.sum(np.square(cfg.Head.spp))*cfg.Head.c2_ouc
cfg.Head.fc1_ins = np.sum(np.square(cfg.Head.spp))*cfg.BaBo.i2_ouc*4
cfg.Head.fc1_ous = 1024
cfg.Head.fc2_ous = 1024
cfg.Head.classes = 2  # in detection also do not contain background
cfg.Head.regression = 4
cfg.Head.class_names = ["coke", "orange"]


cfg.generate.mean = [0.02505453476347024, 0.03179885564375693, 0.0442095491861419]
cfg.generate.std = [0.01424329199190097, 0.029395047427766323, 0.04076346776268015]

cfg.fgenerate.mean = [0.6078857, 0.4183549, 0.51355064]
cfg.fgenerate.std = [0.32497573, 0.27192777, 0.36541626]

cfg.anchor.allowed_border = 50
cfg.anchor.src_info = 224
cfg.anchor.feature_map_size = 56
# cfg.anchor.feat_stride = 4


