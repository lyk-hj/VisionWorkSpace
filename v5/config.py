from easydict import EasyDict as edict
import numpy as np
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

cfg.Cls.c0_inc = 3
cfg.Cls.c1_inc = 8
cfg.Cls.c1_ouc = 16
cfg.Cls.c2_ouc = 32
cfg.Cls.c3_ouc = 64
cfg.Cls.c4_ouc = 64
cfg.Cls.c5_inc = 256
cfg.Cls.c5_ouc = 32
cfg.Cls.classes = 9

cfg.MTCls.c1_inc = 1
cfg.MTCls.c1_ouc = 16
cfg.MTCls.c2_ouc = 32
cfg.MTCls.c3_ouc = 64
cfg.MTCls.c4_ouc = 64
cfg.MTCls.c5_inc = 256
cfg.MTCls.c5_ouc = 32
cfg.MTCls.classes = 8

cfg.BaBo.c0_inc = 1
cfg.BaBo.c1_inc = 4
cfg.BaBo.c1_ouc = 16
cfg.BaBo.c2_ouc = 32
cfg.BaBo.c3_ouc = 64
cfg.BaBo.c4_ouc = 128
cfg.BaBo.i1_ouc = 1024
cfg.BaBo.i2_ouc = 4096

cfg.RPN.c1_ouc = 512
cfg.RPN.anchors = 9
cfg.RPN.cls_ouc = 2*cfg.RPN.anchors
cfg.RPN.reg_ouc = 4*cfg.RPN.anchors

cfg.Head.c1_inc = cfg.BaBo.i2_ouc
cfg.Head.c1_ouc = 1024
cfg.Head.c2_ouc = 256
cfg.Head.spp = [1, 3, 6]
cfg.Head.fc1_ins = np.sum(np.square(cfg.Head.spp))*cfg.Head.c2_ouc
cfg.Head.fc1_ous = 4096
cfg.Head.fc2_ous = 1024
cfg.Head.classes = 3


cfg.generate.mean = [0.024906645, 0.03563779, 0.030372731]
cfg.generate.std = [0.046602514, 0.051798645, 0.049145553]


