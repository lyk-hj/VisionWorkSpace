import torch
from torch import nn
from torchstat import stat
from torch.nn.common_types import _size_2_t
from config import cfg
from onnx_opcounter import calculate_params


# from hj_generate_v4_5 import classes

# c0_inc = 1
# c1_inc = 3
# c1_ouc = 16
# c2_ouc = 32
# c3_ouc = 64
# c4_ouc = 64
# c5_inc = 256
# c5_ouc = 32
# classes = 9


class Conv(nn.Module):
    # dropout should not impact in convolutional layer
    def __init__(self, inc, ouc, k, s=1, p: _size_2_t = 0, g=1, act=True, bn=True) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc,
                              out_channels=ouc,
                              kernel_size=k,
                              stride=s,
                              padding=p,
                              groups=g,
                              bias=False)
        self.bn = nn.BatchNorm2d(ouc) if bn else nn.Identity()  # Identity do not place any computation and memory
        # if isinstance(act, bool):
        self.act = nn.LeakyReLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # print(self.act)

    def forward(self, x):
        return self.bn(self.act(self.conv(x)))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):  # B,C,H,W
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SPPNet(nn.Module):
    def __init__(self, sizes: list):
        super(SPPNet, self).__init__()
        self.spp = [nn.AdaptiveMaxPool2d(i) for i in sizes]  # overlap when cannot exact division
        self.flatten = nn.Flatten()
        # print(self.spp)

    def forward(self, x):
        x = [self.flatten(s(x)) for s in self.spp]
        return torch.cat(x, 1)


class BottleNeck(nn.Module):
    #  do not change the scale, just alter the channel
    def __init__(self, inc, ouc, dw=True) -> None:
        super(BottleNeck, self).__init__()
        self.bottle_neck = nn.Sequential(
            Conv(inc=inc, ouc=inc // 2, k=1, bn=False),
            Conv(inc=inc // 2, ouc=inc, k=3, p=1, g=inc // 2 if dw else 1, bn=False),
            Conv(inc=inc, ouc=ouc, k=1, g=inc if dw else 1),
        )

    def forward(self, x):
        return self.bottle_neck(x)


class FC(nn.Module):
    def __init__(self, ins, ous, bias=False, drop=False, act=False, bn=True, inplace=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features=ins,
                            out_features=ous,
                            bias=bias)
        self.dropout = nn.Dropout(0.5 if drop else 0, inplace=inplace)
        self.act = nn.ReLU(inplace=inplace) if act else nn.Identity()
        self.bn = nn.BatchNorm1d(ous) if bn else nn.Identity()

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.fc(x))))


class InceptionV0(nn.Module):
    def __init__(self, inc, ouc, dw=True):
        #  also do not change the scale, just alter the channel
        super(InceptionV0, self).__init__()
        self.conv1 = Conv(inc=inc, ouc=ouc, k=1)
        self.conv2 = Conv(inc=inc, ouc=ouc, k=3, p=1, g=inc if dw else 1)
        self.conv3 = Conv(inc=inc, ouc=ouc, k=5, p=2, g=inc if dw else 1)
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.BatchNorm2d(inc),  # use for pool
            nn.ReLU(inplace=True)
        )
        self.concat = Concat()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        p = self.pool(x)
        return self.concat([c1, c2, c3, p])


class InceptionV1(nn.Module):
    def __init__(self, inc, ouc, dw=True):
        #  also do not change the scale, just alter the channel
        super(InceptionV1, self).__init__()
        self.conv1 = Conv(inc=inc, ouc=ouc, k=1)
        self.conv2 = nn.Sequential(
            Conv(inc=inc, ouc=inc // 4, k=1, bn=False),  # dimension reduction
            Conv(inc=inc // 4, ouc=ouc, k=3, p=1, g=inc // 4 if dw else 1)
        )
        self.conv3 = nn.Sequential(
            Conv(inc=inc, ouc=inc // 4, k=1, bn=False),
            Conv(inc=inc // 4, ouc=ouc, k=5, p=2, g=inc // 4 if dw else 1)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv(inc=inc, ouc=ouc, k=1, g=inc if dw else 1)
        )
        self.concat = Concat()

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        p = self.pool(x)
        return self.concat([c1, c2, c3, p])


# series
class SInceptionV2(nn.Module):
    def __init__(self, inc, ouc, version, dw=True):
        #  also do not change the scale, just alter the channel
        super(SInceptionV2, self).__init__()
        # conv1 only one version
        self.conv1 = Conv(inc=inc, ouc=ouc, k=1)

        # all version of conv2 in InceptionV2
        self.conv2_1 = nn.Sequential(
            Conv(inc=inc, ouc=inc // 4, k=1, bn=False),  # dimension reduction
            Conv(inc=inc // 4, ouc=ouc, k=3, p=1, g=inc // 4 if dw else 1)
        )
        self.conv2_2 = nn.Sequential(
            Conv(inc=inc, ouc=inc // 4, k=1, bn=False),  # dimension reduction
            Conv(inc=inc // 4, ouc=ouc, k=(3, 1), p=(1, 0), g=inc // 4 if dw else 1),
            Conv(inc=ouc, ouc=ouc, k=(1, 3), p=(0, 1), g=ouc if dw else 1)
        )
        # self.conv2_3 = nn.Sequential(
        #     Conv(inc=inc, ouc=inc / 4, k=1, bn=False),  # dimension reduction
        #     Conv(inc=inc / 4, ouc=ouc, k=3, p=1, g=inc / 4 if dw else 1),
        # )

        # all version of conv3 in InceptionV3
        self.conv3_1 = nn.Sequential(
            Conv(inc=inc, ouc=inc // 4, k=1, bn=False),
            Conv(inc=inc // 4, ouc=ouc, k=3, p=1, g=inc // 4 if dw else 1, bn=False),
            Conv(inc=ouc, ouc=ouc, k=3, p=1, g=ouc if dw else 1)
        )
        self.conv3_2 = nn.Sequential(
            Conv(inc=inc, ouc=inc // 4, k=1, bn=False),
            Conv(inc=inc // 4, ouc=ouc // 2, k=(3, 1), p=(1, 0), g=inc // 4 if dw else 1, bn=False),
            Conv(inc=ouc // 2, ouc=ouc // 2, k=(1, 3), p=(0, 1), g=ouc // 2 if dw else 1, bn=False),
            Conv(inc=ouc // 2, ouc=ouc, k=(3, 1), p=(1, 0), g=ouc // 2 if dw else 1),
            Conv(inc=ouc, ouc=ouc, k=(1, 3), p=(0, 1), g=ouc if dw else 1)
        )

        # pool purely one version
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv(inc=inc, ouc=ouc, k=1, g=inc if dw else 1)
        )
        self.concat = Concat()

        c2_f = {
            1: self.conv2_1,
            2: self.conv2_2
        }

        c3_f = {
            1: self.conv3_1,
            2: self.conv3_2
        }

        # chose need version of InceptionV2
        self.conv2 = c2_f[version]
        self.conv3 = c3_f[version]
        # print(self.conv2)
        # print(self.conv3)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        p = self.pool(x)
        return self.concat([c1, c2, c3, p])


# parallel
class PInceptionV2(nn.Module):
    def __init__(self, inc, ouc, version, dw=True):
        super(PInceptionV2, self).__init__()
        self.conv1 = Conv(inc=inc, ouc=ouc, k=1)

        self.conv2_1 = Conv(inc=inc, ouc=inc // 4, k=1, bn=False)  # dimension reduction
        self.conv2_2 = Conv(inc=inc // 4, ouc=ouc // 2, k=(3, 1), p=(1, 0), g=inc // 4 if dw else 1)
        self.conv2_3 = Conv(inc=inc // 4, ouc=ouc // 2, k=(1, 3), p=(0, 1), g=inc // 4 if dw else 1)

        self.conv3_1 = None
        if version == 1:
            self.conv3_1 = nn.Sequential(
                Conv(inc=inc, ouc=inc // 4, k=1, bn=False),
                Conv(inc=inc // 4, ouc=ouc // 2, k=3, p=1, g=inc // 4 if dw else 1, bn=False),
            )
        else:
            self.conv3_1 = nn.Sequential(
                Conv(inc=inc, ouc=inc // 4, k=1, bn=False),
                Conv(inc=inc // 4, ouc=ouc // 2, k=(3, 1), p=(1, 0), g=inc // 4 if dw else 1, bn=False),
                Conv(inc=ouc // 2, ouc=ouc // 2, k=(1, 3), p=(0, 1), g=ouc // 2 if dw else 1, bn=False),
            )

        self.conv3_2 = Conv(inc=ouc // 2, ouc=ouc // 2, k=(3, 1), p=(1, 0), g=ouc // 2 if dw else 1)
        self.conv3_3 = Conv(inc=ouc // 2, ouc=ouc // 2, k=(1, 3), p=(0, 1), g=ouc // 2 if dw else 1)

        # pool purely one version
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv(inc=inc, ouc=ouc, k=1, g=inc if dw else 1)
        )
        self.concat = Concat()
        # print(self.conv3_1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2_1 = self.conv2_1(x)
        c2 = self.concat([self.conv2_2(c2_1), self.conv2_3(c2_1)])
        c3_1 = self.conv3_1(x)
        c3 = self.concat([self.conv3_2(c3_1), self.conv3_3(c3_1)])
        p = self.pool(x)
        return self.concat([c1, c2, c3, p])


class InceptionResNet(nn.Module):
    def __init__(self):
        super(InceptionResNet, self).__init__()
        pass

    def forward(self, x):
        pass


class ClsModel(nn.Module):
    def __init__(self) -> None:
        super(ClsModel, self).__init__()
        # extend module in class will increase the memory of pytorch model file, but do not function in onnx
        # self.bn = nn.BatchNorm2d(cfg.Cls.c0_inc)
        self.conv0 = Conv(inc=cfg.Cls.c0_inc, ouc=cfg.Cls.c1_inc, k=3, p=1)
        self.bottle_neck1 = BottleNeck(inc=cfg.Cls.c1_inc, ouc=cfg.Cls.c1_ouc)  # shortcut conjunction
        self.conv1 = Conv(inc=cfg.Cls.c1_inc, ouc=cfg.Cls.c1_ouc, k=1, g=cfg.Cls.c1_inc)  # dimension up
        # Max pooling 24->12
        self.bottle_neck2 = BottleNeck(inc=cfg.Cls.c1_ouc, ouc=cfg.Cls.c2_ouc)
        self.conv2 = Conv(inc=cfg.Cls.c1_ouc, ouc=cfg.Cls.c2_ouc, k=1, g=cfg.Cls.c1_ouc)
        # Max pooling 12->6
        self.bottle_neck3 = BottleNeck(inc=cfg.Cls.c2_ouc, ouc=cfg.Cls.c3_ouc)
        self.conv3 = Conv(inc=cfg.Cls.c2_ouc, ouc=cfg.Cls.c3_ouc, k=1, g=cfg.Cls.c2_ouc)
        # Inception
        self.inception = SInceptionV2(inc=cfg.Cls.c3_ouc, ouc=cfg.Cls.c4_ouc, version=1)
        self.concat = Concat()
        # Max pooling 6->3
        # AdaptiveMaxPool2d是自适应kernel，故动态尺寸输入时会出现输出0尺寸，
        # AdaptiveAveragePool2d是对输入求平均即可，故不会为0尺寸输出，即使0尺寸也会为1尺寸输出
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),  # it is something like the SPPNet, SPPNet is not indispensable for this model
            Conv(inc=cfg.Cls.c5_inc, ouc=cfg.Cls.c5_ouc, k=1),
            nn.Flatten(),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=cfg.Cls.c5_ouc, out_features=cfg.Cls.fclasses, bias=False),
            nn.Softmax(1),
        )
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        # x = self.conv0(self.bn(x))
        x = self.conv0(x)
        hid1 = self.max_pool(self.bottle_neck1(x) + self.conv1(x))
        hid2 = self.max_pool(self.bottle_neck2(hid1) + self.conv2(hid1))
        hid3 = self.bottle_neck3(hid2) + self.conv3(hid2)
        hid4 = self.max_pool(self.inception(hid3))
        return self.dense(hid4)


class MTClsModel(nn.Module):
    def __init__(self) -> None:
        super(MTClsModel, self).__init__()
        # extend module in class will increase the memory of pytorch model file, but do not function in onnx
        self.bottle_neck1 = BottleNeck(inc=cfg.MTCls.c1_inc, ouc=cfg.MTCls.c1_ouc)  # shortcut conjunction
        self.conv1 = Conv(inc=cfg.MTCls.c1_inc, ouc=cfg.MTCls.c1_ouc, k=1)  # dimension up
        # Max pooling 24->12
        self.bottle_neck2 = BottleNeck(inc=cfg.MTCls.c1_ouc, ouc=cfg.MTCls.c2_ouc)
        self.conv2 = Conv(inc=cfg.MTCls.c1_ouc, ouc=cfg.MTCls.c2_ouc, k=1)
        # Max pooling 12->6
        self.bottle_neck3 = BottleNeck(inc=cfg.MTCls.c2_ouc, ouc=cfg.MTCls.c3_ouc)
        self.conv3 = Conv(inc=cfg.MTCls.c2_ouc, ouc=cfg.MTCls.c3_ouc, k=1, g=cfg.MTCls.c2_ouc)
        # Inception
        self.conv4_1 = Conv(inc=cfg.MTCls.c3_ouc, ouc=cfg.MTCls.c4_ouc, k=1, g=cfg.MTCls.c3_ouc)
        self.conv4_2 = Conv(inc=cfg.MTCls.c3_ouc, ouc=cfg.MTCls.c4_ouc, k=3, p=1, g=cfg.MTCls.c3_ouc)
        self.conv4_3 = Conv(inc=cfg.MTCls.c3_ouc, ouc=cfg.MTCls.c4_ouc, k=5, p=2, g=cfg.MTCls.c3_ouc)
        # self.conv4_4 = BlockNeck(inc=c3_ouc,ouc=c4_ouc,drop=True)  # 即使没有调用，pt文件也会保存这一层结构, 但onnx不会
        # self.concat = Concat()
        # Max pooling 6->3
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            Conv(inc=cfg.MTCls.c5_inc, ouc=cfg.MTCls.c5_ouc, k=1),
            nn.Flatten(),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=cfg.MTCls.c5_ouc, out_features=cfg.MTCls.classes + 2, bias=False),
        )
        self.softmax = nn.Softmax(1)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        hid1 = self.max_pool(self.bottle_neck1(x) + self.conv1(x))
        hid2 = self.max_pool(self.bottle_neck2(hid1) + self.conv2(hid1))
        hid3 = self.bottle_neck3(hid2) + self.conv3(hid2)
        hid4 = self.max_pool(torch.cat((hid3, self.conv4_1(hid3), self.conv4_2(hid3), self.conv4_3(hid3)), dim=1))
        fc = self.dense(hid4)
        output = self.concat(
            [self.softmax(fc[:, :2]), self.softmax(fc[:, 2:cfg.MTCls.classes + 2])])  # ends should point out
        return output


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.conv0 = Conv(inc=cfg.BaBo.c0_inc, ouc=cfg.BaBo.c1_inc, k=3, p=1)
        self.bottle_neck1 = BottleNeck(inc=cfg.BaBo.c1_inc, ouc=cfg.BaBo.c1_ouc, dw=False)  # shortcut conjunction
        self.conv1 = Conv(inc=cfg.BaBo.c1_inc, ouc=cfg.BaBo.c1_ouc, k=1)  # dimension up
        self.bottle_neck2 = BottleNeck(inc=cfg.BaBo.c1_ouc, ouc=cfg.BaBo.c2_ouc, dw=False)
        self.conv2 = Conv(inc=cfg.BaBo.c1_ouc, ouc=cfg.BaBo.c2_ouc, k=1)

        self.bottle_neck3 = BottleNeck(inc=cfg.BaBo.c2_ouc, ouc=cfg.BaBo.c3_ouc)
        self.conv3 = Conv(inc=cfg.BaBo.c2_ouc, ouc=cfg.BaBo.c3_ouc, k=1)
        self.bottle_neck4 = BottleNeck(inc=cfg.BaBo.c3_ouc, ouc=cfg.BaBo.c4_ouc)
        self.conv4 = Conv(inc=cfg.BaBo.c3_ouc, ouc=cfg.BaBo.c4_ouc, k=1)
        # Max pooling size/2
        self.inception1 = SInceptionV2(inc=cfg.BaBo.c4_ouc, ouc=cfg.BaBo.i1_ouc, version=1)  # real output is i1_ouc*4

        self.inception2 = SInceptionV2(inc=cfg.BaBo.i1_ouc * 4, ouc=cfg.BaBo.i2_ouc,
                                       version=2)  # real output is i2_ouc*4
        # Max pooling size/2
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        c0 = self.conv0(x)
        re1 = self.bottle_neck1(c0) + self.conv1(c0)
        re2 = self.bottle_neck2(re1) + self.conv2(re1)
        re3 = self.bottle_neck3(re2) + self.conv3(re2)
        re4 = self.max_pool(self.bottle_neck4(re3) + self.conv4(re3))
        in1 = self.inception1(re4)
        in2 = self.max_pool(self.inception2(in1))
        return in2


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.feature_extractor = BackBone()
        self.conv1 = Conv(inc=cfg.BaBo.i2_ouc * 4, ouc=cfg.RPN.c1_ouc, k=3, p=1)
        # self.task = None
        # if task == 1:
        self.classification = Conv(inc=cfg.RPN.c1_ouc, ouc=cfg.RPN.cls_ouc, k=1, act=False, bn=False)
        # else:
        self.regression = Conv(inc=cfg.RPN.c1_ouc, ouc=cfg.RPN.reg_ouc, k=1, act=nn.Tanh(), bn=False)

    def forward(self, x, sharing=False):
        feature_map = None
        if sharing:
            feature_map = x
        else:
            feature_map = self.feature_extractor(x)
        c1 = self.conv1(feature_map)
        class_map = self.classification(c1)
        regress_map = self.regression(c1)
        return class_map, regress_map


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.feature_extractor = BackBone()
        self.spp = SPPNet(cfg.Head.spp)
        self.dense = nn.Sequential(
            nn.Dropout(inplace=True),
            FC(ins=cfg.Head.fc1_ins, ous=cfg.Head.fc1_ous, drop=True, act=True, bn=True),
            FC(ins=cfg.Head.fc1_ous, ous=cfg.Head.fc2_ous, drop=True, act=True, bn=True),
            FC(ins=cfg.Head.fc2_ous, ous=cfg.Head.classes + 1 + cfg.Head.regression, bn=False),
            nn.Tanh()
        )

    def forward(self, x, proposals, sharing=False):
        predictions = []
        feature_map = None
        if sharing:
            feature_map = x
        else:
            feature_map = self.feature_extractor(x)
        roi_pool_map = None
        # from IPython import embed;embed()
        for proposal in proposals:
            step = cfg.anchor.src_info // feature_map.shape[2]
            # print(x.shape[0])
            map_position = [int(proposal[0] // step), int(proposal[1] // step),
                            int(proposal[2] // step), int(proposal[3] // step)]  # various scale so no multiple batch
            roi_pool = self.spp(feature_map[..., map_position[0]:map_position[2],
                                map_position[1]:map_position[3]])
            if roi_pool_map is None:
                roi_pool_map = roi_pool
            else:
                roi_pool_map = torch.cat([roi_pool_map, roi_pool], dim=0)
        # print(roi_pool_map.shape)
        output = self.dense(roi_pool_map)
        # predictions.append(output)
        return output


if __name__ == "__main__":
    model = Head()
    # file_name = '2023_4_19_hj_num_2'
    # model_path = '../weight/' + file_name + '.pt'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.load(model_path, map_location='cpu')
    # print(torch.reshape(model.dense[1].conv.weight, torch.Size([32, 256])))
    stat(model, (3, 224, 224))
    #
    # model = Conv(inc=3, ouc=4, k=3, s=1, act=nn.Sigmoid())

