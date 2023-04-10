import glob
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import cv2

# class_label = [1,2,3,4,5]

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.
])

#通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
# 类初始化
    def __init__(self, root, labels, transform):
        self.imgs_path = root
        self.labels = labels
        self.transforms = transform

# 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = self.labels[index]
        src = cv2.imread(img_path)
        data = cv2.resize(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY),(20,30))
        data = data.astype(np.float32) / 255.0
        return data,label
# 返回长度
    def __len__(self):
        return len(self.imgs_path)
# def data_generate():
    #使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'../data2/*.jpg')
check_imgs_path = glob.glob(r'../data3/*.jpg')
# for var in all_imgs_path:
#     print(var)

# hjNum_dataset = Mydataset(all_imgs_path)
# hjNum_dataloader = torch.utils.data.DataLoader(hjNum_dataset,batch_size=5)
# print(next(iter(hjNum_dataloader)))

all_labels = []
check_labels = []
class_id=0
for img in all_imgs_path:
    class_id = int(img[9])
    if (img[8] == '6'):
        class_id = 5
    all_labels.append(class_id)

for img in check_imgs_path:
    class_id = int(img[9])
    if (img[8] == '6'):
        class_id = 5
    check_labels.append(class_id)
# print(len(all_labels))

BATCH_SIZE = 2

# hjNum_dataset = Mydataset(all_imgs_path,all_labels,transform)
# hjNum_dataloader = data.DataLoader(
#     hjNum_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True
# )

index = np.random.permutation(len(all_imgs_path))
all_imgs_path = np.array(all_imgs_path)[index]#打乱数据
all_labels = np.array(all_labels)[index]

a = int(len(all_imgs_path)*0.8)
print(a)

train_imgs = all_imgs_path[:a]
train_labels = all_labels[:a]
test_imgs = all_imgs_path[a:]
test_labels = all_labels[a:]

train_dataset = Mydataset(train_imgs,train_labels,transform)#get train dataset rule
train_dataloader = data.DataLoader(train_dataset,BATCH_SIZE,True,drop_last=True)#get train dataloader

test_dataset = Mydataset(test_imgs,test_labels,transform)#get test dataset rule
test_dataloader = data.DataLoader(test_dataset,BATCH_SIZE,True,drop_last=True)#get test dataloader

check_dataset = Mydataset(check_imgs_path,check_labels,transform)
check_dataloader = data.DataLoader(check_dataset,BATCH_SIZE,True,drop_last=True)

# print(next(iter(train_dataloader)))
# print(next(iter(test_dataloader)))
# print(next(iter(check_dataloader)))

# if __name__ == "__main__":
#     data_generate()
