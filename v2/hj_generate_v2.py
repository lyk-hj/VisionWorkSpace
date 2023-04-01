import glob
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import cv2

crop_element = [[0,0],[3,2],[6,0],[6,4],[0,4],[0,0],[1,1],[3,0],[3,2],[0,2]]

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.
])

#通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
# 类初始化
    def __init__(self, root, crop_flag, transform, is_verification):
        self.imgs_path = root
        self.crop_flag=crop_flag
        self.is_verification = is_verification
        self.transforms = transform

# 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        # print(img_path)
        # print(img_path[8])
        label = int(img_path[9])
        if (img_path[9] == '6'):
            label = 5
        # label = self.labels[index]
        src = cv2.imread(img_path)
        src = cv2.resize(src,(20,30))
        data = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        if not self.is_verification:
            crop_mode = self.crop_flag[index]
            if crop_mode < 5:
                data = cv2.resize(data, (24, 36))
                crop_h_start = crop_element[crop_mode][0]
                crop_w_start = crop_element[crop_mode][1]
                # print(data.size)
                # cv2.imshow("data", data)
                # cv2.waitKey(0)
                data = data[crop_h_start:crop_h_start + 30,
                       crop_w_start:crop_w_start + 20]
                # print(data.size)
                # cv2.imshow("data", data)
                # cv2.waitKey(0)
            elif crop_mode == 5:
                data = cv2.resize(data, (20,30))
                # cv2.imshow("data", data)
                # cv2.waitKey(0)
            elif crop_mode > 5:
                data = cv2.resize(data, (22,33))
                crop_h_start = crop_element[crop_mode-1][0]
                crop_w_start = crop_element[crop_mode-1][1]
                data = data[crop_h_start:crop_h_start + 30,
                       crop_w_start:crop_w_start + 20]
                # cv2.imshow("data", data)
                # cv2.waitKey(0)
        data = np.reshape(data.astype(np.float32) / 255.0, (1,30,20))
        # print(data.shape)
        return data,label
# 返回长度
    def __len__(self):
        return len(self.imgs_path)

#使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'../data2/*.jpg')
check_imgs_path = glob.glob(r'../data3/*.jpg')
extra_imgs_path = glob.glob(r"../data1/*.jpg")
# print(extra_imgs_path)

# stronger train data
strong_data_path=[]
data_crop_flags=[]
for i in range(11):
    for img in all_imgs_path:
        strong_data_path.append(img)
        data_crop_flags.append(i)
    for img in extra_imgs_path:
        strong_data_path.append(img)
        data_crop_flags.append(i)

# all_labels = []
# check_labels = []
# class_id=0
# for img in strong_data_path:
#     class_id = int(img[8])
#     if (img[8] == '6'):
#         class_id = 5
#     all_labels.append(class_id)
#
# for img in check_imgs_path:
#     class_id = int(img[8])
#     if (img[8] == '6'):
#         class_id = 5
#     check_labels.append(class_id)
# print(len(all_labels))

BATCH_SIZE = 2

index = np.random.permutation(len(strong_data_path))
all_imgs_path = np.array(strong_data_path)[index]#打乱数据
# all_labels = np.array(all_labels)[index]
all_crop_flags = np.array(data_crop_flags)[index]

a = int(len(strong_data_path)*0.8)
print(a)
train_imgs = all_imgs_path[:a]
# train_labels = all_labels[:a]
train_crop_flags = all_crop_flags[:a]
test_imgs = all_imgs_path[a:]
# test_labels = all_labels[a:]
test_crop_flags = all_crop_flags[a:]

train_dataset = Mydataset(train_imgs,train_crop_flags,transform,is_verification=False)#get train dataset rule
train_dataloader = data.DataLoader(train_dataset,BATCH_SIZE,True,drop_last=True)#get train dataloader

test_dataset = Mydataset(test_imgs,test_crop_flags,transform,is_verification=False)#get test dataset rule
test_dataloader = data.DataLoader(test_dataset,BATCH_SIZE,True,drop_last=True)#get test dataloader

check_dataset = Mydataset(check_imgs_path,[],transform,is_verification=True)
check_dataloader = data.DataLoader(check_dataset,BATCH_SIZE,True,drop_last=True)

print(next(iter(train_dataloader)))
print(next(iter(test_dataloader)))
print(next(iter(check_dataloader)))

# if __name__ == "__main__":
#     data_generate()