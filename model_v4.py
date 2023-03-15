import torch
from torch import nn

c1_inc = 1
c1_ouc = 16
c2_ouc = 32
c3_ouc = 64
c4_ouc = 64
c5_inc = 256
c5_ouc = 32
classes = 9


class Conv(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, p=0, g=1, act=True, drop=False) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=inc,
                                 out_channels=ouc,
                                 kernel_size=k,
                                 stride=s,
                                 padding=p,
                                 groups=g,
                                 bias=False)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = nn.ReLU() if act else nn.Identity()
        self.dropout = nn.Dropout(0.5 if drop else 0)

    def forward(self, x):
        return self.act(self.bn(self.dropout(self.conv(x))))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):  # B,C,H,W
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class BlockNeck(nn.Module):
    #  do not change the scale, just alter the channel
    def __init__(self, inc, ouc, s=1, g=1, act=True, drop=False) -> None:
        super(BlockNeck, self).__init__()
        self.block_neck = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=1,
                      stride=s,
                      groups=inc,
                      bias=False,
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=3,
                      stride=s,
                      padding=1,
                      groups=inc,
                      bias=False
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=ouc,
                      kernel_size=1,
                      stride=s,
                      groups=g,
                      bias=False
                      ),
        )
        self.act = nn.ReLU() if act else nn.Identity()
        self.dropout = nn.Dropout(0.5 if drop else 0)
        self.bn = nn.BatchNorm2d(ouc)

    def forward(self, x):
        return self.act(self.bn(self.dropout(self.block_neck(x))))


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        # extend module in class will increase the memory of pytorch model file, but do not function in onnx
        self.block_neck1 = BlockNeck(inc=c1_inc,ouc=c1_ouc)  # shortcut conjunction
        self.conv1 = Conv(inc=c1_inc,ouc=c1_ouc,k=1)  # dimension up
        # Max pooling 24->12
        self.block_neck2 = BlockNeck(inc=c1_ouc,ouc=c2_ouc)
        self.conv2 = Conv(inc=c1_ouc,ouc=c2_ouc,k=1)
        # Max pooling 12->6
        self.block_neck3 = BlockNeck(inc=c2_ouc,ouc=c3_ouc,drop=True)
        self.conv3 = Conv(inc=c2_ouc,ouc=c3_ouc,k=1,g=c2_ouc,drop=True)
        # Inception
        self.conv4_1 = Conv(inc=c3_ouc,ouc=c4_ouc,k=1,g=c3_ouc)
        self.conv4_2 = Conv(inc=c3_ouc,ouc=c4_ouc,k=3,p=1,g=c3_ouc,drop=True)
        self.conv4_3 = Conv(inc=c3_ouc,ouc=c4_ouc,k=5,p=2,g=c3_ouc,drop=True)
        self.conv4_4 = BlockNeck(inc=c3_ouc,ouc=c4_ouc,drop=True)
        self.concat = Concat()
        # Max pooling
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(inc=c5_inc, ouc=c5_ouc, k=1, drop=True),
            nn.Flatten(),
            nn.Linear(in_features=c5_ouc, out_features=classes, bias=False),
            nn.Softmax(1),
        )
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        hid1 = self.max_pool(self.block_neck1(x) + self.conv1(x))
        hid2 = self.max_pool(self.block_neck2(hid1) + self.conv2(hid1))
        hid3 = self.block_neck3(hid2) + self.conv3(hid2)
        hid4 = self.max_pool(self.concat((hid3, self.conv4_1(hid3), self.conv4_2(hid3), self.conv4_3(hid3))))
        return self.dense(hid4)

if __name__ == "__main__":
    model = Model()
    print(model)

