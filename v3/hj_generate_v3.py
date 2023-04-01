import glob
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import cv2

# python rules that height before width

crop_element = [[0,0],[2,2],[4,0],[4,4],[0,4],  # 28
                [0,0],[1,1],[2,0],[2,2],[0,2],  # 26
                [0,0],[1,0],[0,1],[1,1],        # 25
                [0,0],[2,1],[1,2],[1,1],[2,2],[3,3],[3,2],[2,3],  # 27
                [0,0]]  # 24

transform = transforms.Compose([
    transforms.ToTensor(),
])

def resize(data, size):
    return cv2.resize(data,size)

def stretch(data, act_size, need_size, crop_mode):
    back = np.zeros(act_size, np.uint8)
    h_size_offset = (act_size[0] - need_size[0]) // 2
    w_size_offset = (act_size[1] - need_size[1]) // 2
    crop_h_start = crop_element[crop_mode][0] + h_size_offset
    crop_w_start = crop_element[crop_mode][1] + w_size_offset
    back[crop_h_start:crop_h_start + 24, crop_w_start:crop_w_start + 24] = data
    return cv2.resize(back, need_size)

def out_of_shape(data, out_size):
    data = resize(data, (out_size[0]*2,out_size[1]*2))
    rows, cols = data.shape[:2]
    max_dim,min_dim = (rows,cols) if rows > cols else (cols, rows)
    # print(max_dim,min_dim)
    back = np.zeros((max_dim,max_dim),np.uint8)
    back[out_size[0]//2:out_size[0]//2+rows,0:cols] = data
    data = resize(back,(min_dim//2,min_dim//2))
    # return data[0:out_size[1],0:out_size[0]]
    return data

def clip_superfluous_pixel(data,label):
    if label == 0:
        return data
    rows,cols = data.shape[:2]
    row_start = 0
    row_end = rows
    col_start = 0
    col_end = cols
    boundary_flag = 0
    for i in range(rows):
        for j in range(cols):
            if boundary_flag == 0:
                if data[i][j] == 255:
                    row_start = i
                    boundary_flag = 1
            elif boundary_flag == 1:
                    break
        if boundary_flag == 1:
            if sum(data[i]) == 0:
                row_end = i
                break
    boundary_flag = 0
    for i in range(cols):
        for j in range(rows):
            if boundary_flag == 0:
                if data[j][i] == 255:
                    col_start = i
                    boundary_flag = 1
            elif boundary_flag == 1:
                break
        if boundary_flag == 1:
            if sum(data[:,i]) == 0:
                col_end = i
                break
    # print(row_start,row_end,col_start,col_end)
    return data[row_start:row_end,col_start:col_end]


#通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
# 类初始化
    def __init__(self, root, crop_flag, transform, is_verification):
        self.imgs_path = root
        self.crop_flag=crop_flag
        self.is_verification = is_verification
        self.transforms = transform

# 返回长度
    def __len__(self):
        return len(self.imgs_path)
# 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = int(img_path[9])
        if (img_path[9] == '6'):
            label = 5
        src = cv2.imread(img_path)
        src = cv2.resize(src,(24,24))
        data = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _,data = cv2.threshold(data,0,255,cv2.THRESH_BINARY or cv2.THRESH_OTSU)
        if not self.is_verification:
            crop_mode = self.crop_flag[index]
            if crop_mode < 5:
                data = clip_superfluous_pixel(data,label)
                data = resize(data, (28, 28))
                # data = out_of_shape(data, (63,28))
                # print(data.size)

            elif crop_mode >= 5 and crop_mode < 10:
                # data = clip_superfluous_pixel(data,label)
                # data = resize(data, (26,26))
                data = stretch(data, (32,26), (26,26), crop_mode)
                # print(data.size)

            elif crop_mode >=10 and crop_mode < 14:
                data = resize(data, (25, 25))
                # data = stretch(data, (41,25), (25,25), crop_mode)
                # print(data.size)

            elif crop_mode >=14 and crop_mode < 18:
                # data = clip_superfluous_pixel(data,label)
                # data = resize(data, (27, 27))
                data = stretch(data, (59,27), (27,27), crop_mode)
                # print(data.size)

            elif crop_mode >=18 and crop_mode < 22:
                data = stretch(data, (41,27), (27,27), crop_mode)

            elif crop_mode == 22:
                # data = resize(data, (24, 24))
                data = stretch(data, (24,30), (24,24), crop_mode)
                # print(data.size)

            crop_h_start = crop_element[crop_mode][0]
            crop_w_start = crop_element[crop_mode][1]
            data = data[crop_h_start:crop_h_start + 24,
                   crop_w_start:crop_w_start + 24]
            # if(label == 5):
            #     cv2.imwrite("../enhance_template/"+ img_path[8:],data)
            #     cv2.imshow("data", data)
            #     cv2.waitKey(0)
        data = np.reshape(data.astype(np.float32) / 255.0, (1,24,24))
        return data,label

#使用glob方法来获取数据图片的所有路径
all_imgs_path = glob.glob(r'../data2/*.jpg')
all_pngs_path = glob.glob(r'../data2/*.png')
valid_imgs_path = glob.glob(r'../data3/*.jpg')
extra_imgs_path = glob.glob(r"../data1/*.jpg")
extra_valid_path = glob.glob(r'../data0/*.jpg')
valid_imgs_path.extend(extra_valid_path)
all_imgs_path.extend(extra_imgs_path)
all_imgs_path.extend(all_pngs_path)
# print(extra_imgs_path)
# print(all_pngs_path)

# stronger train data
strong_data_path=[]
data_crop_flags=[]
for i in range(23):
    for img in all_imgs_path:
        strong_data_path.append(img)
        data_crop_flags.append(i)

BATCH_SIZE = 4

index = np.random.permutation(len(strong_data_path))
all_imgs_path = np.array(strong_data_path)[index]#打乱数据
all_crop_flags = np.array(data_crop_flags)[index]

a = int(len(strong_data_path)*0.8)
print(a)
train_imgs = all_imgs_path[:a]
train_crop_flags = all_crop_flags[:a]
test_imgs = all_imgs_path[a:]
test_crop_flags = all_crop_flags[a:]

train_dataset = Mydataset(train_imgs,train_crop_flags,transform,is_verification=False)#get train dataset rule
train_dataloader = data.DataLoader(train_dataset,BATCH_SIZE,True,drop_last=True)#get train dataloader

test_dataset = Mydataset(test_imgs,test_crop_flags,transform,is_verification=False)#get test dataset rule
test_dataloader = data.DataLoader(test_dataset,BATCH_SIZE,True,drop_last=True)#get test dataloader

valid_dataset = Mydataset(valid_imgs_path,[],transform,is_verification=True)
valid_dataloader = data.DataLoader(valid_dataset,BATCH_SIZE,True,drop_last=True)

print(next(iter(train_dataloader)))
# print(next(iter(test_dataloader)))
# print(next(iter(check_dataloader)))

# if __name__ == "__main__":
#     data_generate()
#
# img = cv2.imread("./data1/1_35.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# _,img = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
# img = clip_superfluous_pixel(img)
# cv2.imshow("img",img)
# cv2.waitKey(0)