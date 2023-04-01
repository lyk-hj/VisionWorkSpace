import torch

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model,self).__init__()
        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.MaxPool2d(2),#output 7*12*16 in python is (16,12,7)
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=(2,1),
                            padding=0),#ouput 5*5*32
            torch.nn.MaxPool2d(5),#ouput 1*1*32
            torch.nn.Flatten(),
        )
        self.bn = torch.nn.BatchNorm1d(64)

        self.dense=torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 6),
            torch.nn.Softmax(1),
        )

    def forward(self,x):
        # x=x.view(-1,1,30,20)#front is rows, back is cols
        x=self.conv(x)
        x=self.bn(x)
        x=self.dense(x)
        return x