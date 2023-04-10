import torch
from torch import nn
from torchstat import stat

c1_inc = 1
c1_ouc = 16
c2_ouc = 16
c3_ouc = 32
c4_inc = 64
c4_ouc = 64
c5_inc = 128
c5_ouc = 32
classes = 6


class Conv(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, p=0, g=1, act=True, drop=False) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=ouc,
                      kernel_size=k,
                      stride=s,
                      padding=p,
                      groups=g,
                      bias=False),
            nn.Dropout(),
        ) if drop else nn.Conv2d(in_channels=inc,
                                 out_channels=ouc,
                                 kernel_size=k,
                                 stride=s,
                                 padding=p,
                                 groups=g,
                                 bias=False)
        self.bn = nn.BatchNorm2d(ouc)
        self.act = nn.ReLU() if act else nn.Identity()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, pool=False):
        return self.act(self.bn(self.conv(x))) if not pool else self.pool(self.act(self.bn(self.conv(x))))


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
        self.block_neck = nn.Sequential(
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=1,
                      stride=s,
                      padding=1,
                      groups=g,
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=inc,
                      kernel_size=3,
                      stride=s,
                      padding=1,
                      groups=g,
                      ),
            nn.Conv2d(in_channels=inc,
                      out_channels=ouc,
                      kernel_size=1,
                      stride=s,
                      padding=1,
                      groups=g,
                      ),
        )
        self.act = nn.ReLU() if act else nn.Identity()
        self.dropout = nn.Dropout(0.5 if drop else 0)
        self.bn = nn.BatchNorm2d(ouc)

    def forward(self,x):
        return self.act(self.bn(self.dropout(self.block_neck(x))))


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv1 = Conv(inc=c1_inc, ouc=c1_ouc)  # in:24*24*1, out:22*22*16
        self.conv2 = Conv(inc=c1_ouc, ouc=c2_ouc, g=c1_ouc)  # in:22*22*16, out:20*20*16
        # max_pooling
        self.conv3_1 = Conv(inc=c2_ouc, ouc=c3_ouc)  # in:10*10*16, out:8*8*32
        self.conv3_2 = Conv(inc=c3_ouc, ouc=c3_ouc, g=c3_ouc, p=1)  # in:8*8*32, out:8*8*32
        self.conv3_3 = Conv(inc=c3_ouc, ouc=c4_inc,p=1, drop=True)  # in:8*8*32, out:8*8*64
        # max_pooling
        self.conv4_1 = Conv(inc=c4_inc, ouc=c4_ouc, g=c4_inc)  # in:4*4*64, out:2*2*64
        self.conv4_2 = Conv(inc=c4_inc, ouc=c4_ouc, g=c4_inc, act=False, drop=True)  # in:4*4*64, out:2*2*64
        self.concat = Concat()
        self.dense = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # in:2*2*128, out:1*1*128
            Conv(inc=c5_inc, ouc=c5_ouc, k=1, drop=True),
            nn.Flatten(),
            nn.Linear(in_features=c5_ouc,out_features=classes,bias=False),
            nn.Softmax(1),
        )

    def forward(self, x):
        # x=x.view(-1,1,24,24)#front is rows, back is cols
        x = self.conv2(self.conv1(x), True)
        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x1)
        x = self.conv3_3(x1+x2,True)
        return self.dense(self.concat((self.conv4_1(x),self.conv4_2(x))))


if __name__ == "__main__":
    model = torch.load('../weight/2023_3_3_hj_num_1.pt').to("cpu")
    print(model)
    stat(model, (1, 30, 22))