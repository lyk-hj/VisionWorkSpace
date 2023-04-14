import glob
import random
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import cv2
import torch.nn.functional as F
from config import cfg

# python rules that height before width, while in cv2.resize is width before height
# input_h = 24
# input_w = 24

# will change the crop mode in v5
crop_element = [[0, 0], [2, 2], [4, 0], [4, 4], [0, 4],  # 28
                [0, 0], [1, 1], [2, 0], [2, 2], [0, 2],  # 26
                [0, 0], [1, 0], [0, 1], [1, 1],  # 25
                [0, 0], [2, 1], [1, 2], [1, 1], [2, 2], [3, 3], [3, 2], [2, 3],  # 27
                [0, 0]]  # 24

transform = transforms.Compose([
    transforms.RandomRotation(degrees=15, expand=True),
    # transforms.Resize((24, 24)),  # 这里的resize不是裁剪，是缩放
    transforms.ToTensor(),
])


# class data_consolidate:
#     def __init__(self, data, label, data_border, enhance_flag, crop_mode):
#         self.function = {
#             1: resize,
#             2: stretch,
#             3: out_of_shape,
#             4: clip_superfluous_pixel,
#             5: increase_noise,
#             6: add_white_boundary
#         }
#         self.crop_mode = crop_mode
#         self.data_border = data_border
#         self.label = label
#
#     def random_data_enhance(self):
#         out_size = None
#         if self.crop_mode < 5:
#             # out_size = (28, 28)
#             out_size = (self.data_border + 4, self.data_border + 4)
#         elif self.crop_mode >= 5 and crop_mode < 10:
#             out_size = (self.data_border + 2, self.data_border + 2)
#         elif self.crop_mode >= 10 and crop_mode < 14:
#             out_size = (self.data_border + 1, self.data_border + 1)
#         elif self.crop_mode >= 14 and self.crop_mode < 22:
#             out_size = (self.data_border + 3, self.data_border + 3)
#         elif self.crop_mode == 22:
#             out_size = (self.data_border, self.data_border)
#
#         if self.label == 0:
#             # stochastic flip
#             data = cv2.flip(data, random.randint(-1, 1)) if random.randint(0, 1) else data
#             # random rotation
#             height, width = data.shape[:2]
#             center = (width / 2, height / 2)
#             M = cv2.getRotationMatrix2D(center, random.uniform(-90, 90), 1.0)
#             data = cv2.warpAffine(data, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#         if enhance_flag > 0:
#
#
#             fun = self.functions[enhance_flag]
#             act_long_border = np.random.randint(data_border + 5,
#                                                 data_border + 15)  # numpy的randint不包含最大值，python自己的randint包含最大值
#             act_size = (out_size[0], act_long_border) if random.randint(0, 1) else (act_long_border, out_size[1])
#             data = fun(data=data, out_size=out_size,
#                        act_size=act_size, label=label, crop_mode=crop_mode)
#
#             return crop(data, (data_border, data_border), crop_mode=crop_mode)
#         else:
#             return resize(data, (data_border, data_border))



def random_data_enhance(data, label, data_border, enhance_flag, crop_mode):
    out_size = None
    if crop_mode < 5:
        # out_size = (28, 28)
        out_size = (data_border + 4, data_border + 4, 3)
    elif crop_mode >= 5 and crop_mode < 10:
        out_size = (data_border + 2, data_border + 2, 3)
    elif crop_mode >= 10 and crop_mode < 14:
        out_size = (data_border + 1, data_border + 1, 3)
    elif crop_mode >= 14 and crop_mode < 22:
        out_size = (data_border + 3, data_border + 3, 3)
    elif crop_mode == 22:
        out_size = (data_border, data_border, 3)

    if label == 0:
        # stochastic flip
        data = cv2.flip(data, random.randint(-1, 1)) if random.randint(0, 1) else data
        # random rotation
        height, width = data.shape[:2]
        center = (width / 2, height / 2)
        M = cv2.getRotationMatrix2D(center, random.uniform(-90, 90), 1.0)
        data = cv2.warpAffine(data, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # packaging all functions to a class in v5
    if enhance_flag > 0:
        functions = {
            1: resize,
            2: stretch,
            3: out_of_shape,
            4: clip_superfluous_pixel,
            5: increase_noise,
            6: add_white_boundary
        }
        # print(enhance_flag)
        fun = functions[enhance_flag]
        act_long_border = np.random.randint(data_border + 5,
                                            data_border + 15)  # numpy的randint不包含最大值，python自己的randint包含最大值
        act_size = (out_size[0], act_long_border, 3) if random.randint(0, 1) else (act_long_border, out_size[1], 3)
        data = fun(data=data, out_size=out_size,
                   act_size=act_size, label=label, crop_mode=crop_mode)

        return crop(data, (data_border, data_border, 3), crop_mode=crop_mode)
    else:
        return resize(data, (data_border, data_border, 3))


def resize(data, out_size, act_size=None, label=None, crop_mode=None):
    return cv2.resize(data, (int(out_size[0]), int(out_size[1])))


def stretch(data, out_size, act_size=None, crop_mode=None, label=None):  # fill not only min edge but also max size
    back = np.zeros(out_size, np.uint8)
    rows, cols = data.shape[0], data.shape[1]
    max_dim, min_dim = (rows, cols) if rows > cols else (cols, rows)
    data = resize(data, (out_size[0] * (cols / max_dim), out_size[0] * (rows / max_dim)))
    h_size_offset = (out_size[0] - data.shape[0]) // 2
    w_size_offset = (out_size[1] - data.shape[1]) // 2
    crop_h_start = h_size_offset
    crop_w_start = w_size_offset
    back[crop_h_start:crop_h_start + data.shape[0], crop_w_start:crop_w_start + data.shape[1]] = data
    return back


def out_of_shape(data, out_size, act_size=None, label=None, crop_mode=None):  # fill the min border
    rows, cols = act_size[1], act_size[0]
    max_dim, min_dim = (rows, cols) if rows > cols else (cols, rows)
    data = resize(data, (out_size[0] * (act_size[0] / max_dim), out_size[0] * (act_size[1] / max_dim)))
    rows, cols = data.shape[0], data.shape[1]
    max_dim2, min_dim2 = (rows, cols) if rows > cols else (cols, rows)
    # print(max_dim,min_dim)
    position = (max_dim2 - min_dim2) // 2
    back = np.zeros(out_size, np.uint8)
    if act_size[0] > act_size[1]:
        back[position:position + rows, 0:cols] = data
    else:
        back[0:rows, position:position + cols] = data
    # data = resize(back, (min_dim // 2, min_dim // 2))
    return back


def add_white_boundary(data, out_size, label=None, act_size=None, crop_mode=None):
    direction = random.randint(0, 1)
    if direction:  # landscape
        _w = random.randint(1, 3)  # white width
        _num = random.randint(0, 2)  # whole size or up size or bottom size
        for w in range(_w):
            l_start = np.random.randint(0, data.shape[1] // 2)
            l_end = np.random.randint(data.shape[1] // 2, data.shape[1])
            for lo in range(l_start, l_end):
                if _num == 0:
                    data[w, lo] = [128, 128, 128]
                    data[data.shape[0] - 1 - w, lo] = [128, 128, 128]
                elif _num == 1:
                    data[w, lo] = [128, 128, 128]
                elif _num == 2:
                    data[data.shape[0] - 1 - w, lo] = [128, 128, 128]
    else:  # portrait
        _w = random.randint(1, 4)  # white width
        _num = random.randint(0, 2)  # whole size or left size or right size
        for w in range(_w):
            l_start = np.random.randint(0, data.shape[0] // 2)
            l_end = np.random.randint(data.shape[0] // 2, data.shape[0])
            for lo in range(l_start, l_end):
                if _num == 0:
                    data[lo, w] = [128, 128, 128]
                    data[lo, data.shape[1] - 1 - w] = [128, 128, 128]
                elif _num == 1:
                    data[lo, w] = [128, 128, 128]
                elif _num == 2:
                    data[lo, data.shape[1] - 1 - w] = [128, 128, 128]
    return resize(data, out_size)


def increase_noise(data, out_size, label=None, act_size=None, crop_mode=None):
    height = data.shape[0]
    weight = data.shape[1]

    for i in range(height):
        for j in range(weight):
            num = random.randint(0, 1000)
            if (num > 996):
                data[i, j] = [0, 0, 0]
            elif (num < 4):
                data[i, j] = [128, 128, 128]
    return resize(data, out_size)



def clip_superfluous_pixel(data, out_size, label=None, act_size=None, crop_mode=None):
    if label == 0:
        return resize(data, out_size)
    rows, cols = data.shape[:2]
    row_start = 0
    row_end = rows
    col_start = 0
    col_end = cols
    boundary_flag = 0
    for i in range(rows):
        for j in range(cols):
            if boundary_flag == 0:
                if np.mean(data[i][j]) > 80:
                    row_start = i
                    boundary_flag = 1
            elif boundary_flag == 1:
                break
        if boundary_flag == 1:
            if np.mean(np.mean(data[i], axis=0)) < 50:
                row_end = i
                break
    boundary_flag = 0
    for i in range(cols):
        for j in range(rows):
            if boundary_flag == 0:
                if np.mean(data[j][i]) > 80:
                    col_start = i
                    boundary_flag = 1
            elif boundary_flag == 1:
                break
        if boundary_flag == 1:
            if np.mean(np.mean(data[:, i], axis=0)) < 50:
                col_end = i
                break
    row_end = row_end if row_end >= data.shape[0] / 2 else rows
    col_end = col_end if col_end >= data.shape[1] / 2 else cols
    # print(row_start,row_end,col_start,col_end)
    return resize(data[row_start:row_end, col_start:col_end], out_size)


def crop(data, out_size, label=None, act_size=None, crop_mode=None):
    crop_h_start = crop_element[crop_mode][0]
    crop_w_start = crop_element[crop_mode][1]
    return data[crop_h_start:crop_h_start + out_size[0],
           crop_w_start:crop_w_start + out_size[1]]


# 通过创建data.Dataset子类Mydataset来创建输入
# classes = 8
class Mydataset(data.Dataset):
    # 类初始化
    def __init__(self, root, crop_flag, transform, is_verification):
        self.imgs_path = root
        self.crop_flag = crop_flag
        self.is_verification = is_verification
        self.transforms = transform
        self.data_border = random.randint(24, 30)
        # self.data_border = 24
        print(self.data_border, end="\t")

    # 返回长度
    def __len__(self):
        return len(self.imgs_path)

    # 进行切片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = int(img_path[9])
        data = cv2.imread(img_path)
        # cv2.imshow("src", data)
        if not self.is_verification:
            crop_mode = self.crop_flag[index]
            enhance_flag = random.randint(-5, 5)
            data = random_data_enhance(data, label, self.data_border, enhance_flag, crop_mode)

        else:
            data = resize(data, (self.data_border, self.data_border))
        mask = np.zeros(data.shape, dtype=np.uint8)
        mask = mask + 7
        data = cv2.divide(data, mask)
        # cv2.imshow("data", data)
        # cv2.waitKey(0)
        # data = data.astype(np.float32) / 255.0
        data = (data.astype(np.float32) / 255.0 - cfg.generate.mean) / cfg.generate.std
        data = (np.reshape(data.astype(np.float32), (3, data.shape[0], data.shape[1])))
        # print(data.shape)
        return data, label


# 使用glob方法来获取数据图片的所有路径
all_imgs_path = []
valid_imgs_path = []
data0_path = glob.glob(r'../data0/*g')
data1_path = glob.glob(r"../data1/*g")
# data2_path = glob.glob(r'../data2/*g')
# data1_path = glob.glob(r"../data1/7*")
data2_path = glob.glob(r'../data2/7*')
# data1_path.extend(glob.glob(r"../data1/8*"))
data2_path.extend(glob.glob(r"../data2/8*"))
# data1_path.extend(glob.glob(r"../data1/6*"))
data2_path.extend(glob.glob(r"../data2/6*"))
data2_path.extend(glob.glob(r"../data2/0*"))
data3_path = glob.glob(r'../data3/*g')
data7_path = glob.glob(r"../data7/*g")

# valid_imgs_path.extend(data0_path)
# valid_imgs_path.extend(data3_path)
valid_imgs_path.extend(data7_path)
# print(valid_imgs_path)

all_imgs_path.extend(data1_path)
all_imgs_path.extend(data2_path)
# all_imgs_path.extend(data0_path)

# stronger train data
strong_data_path = []
data_crop_flags = []
for i in range(23):
    for img in all_imgs_path:
        strong_data_path.append(img)
        data_crop_flags.append(i)

BATCH_SIZE = 16

index = np.random.permutation(len(strong_data_path))
all_imgs_path = np.array(strong_data_path)[index]  # 打乱数据
all_crop_flags = np.array(data_crop_flags)[index]

a = int(len(strong_data_path) * 0.8)
print(a)
train_imgs = all_imgs_path[:a]
train_crop_flags = all_crop_flags[:a]
test_imgs = all_imgs_path[a:]
test_crop_flags = all_crop_flags[a:]

train_datasets = [Mydataset(train_imgs, train_crop_flags, transform, is_verification=False) for _ in range(10)]  # get train dataset rule
train_dataloaders = [data.DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True) for train_dataset in train_datasets]  # get train dataloader

test_dataset = Mydataset(test_imgs, test_crop_flags, transform, is_verification=False)  # get test dataset rule
test_dataloader = data.DataLoader(test_dataset, BATCH_SIZE, True, drop_last=True)  # get test dataloader

valid_dataset = Mydataset(valid_imgs_path, [], transform, is_verification=True)
valid_dataloader = data.DataLoader(valid_dataset, BATCH_SIZE, True, drop_last=True)


def get_mean_std(dataset):
    r = []
    g = []
    b = []
    for i, j in dataset:
        r.append(i[0])
        g.append(i[1])
        b.append(i[2])
    r_mean = np.mean(r, axis=0)
    g_mean = np.mean(g, axis=0)
    b_mean = np.mean(b, axis=0)
    r_std = np.std(r, ddof=1, axis=0)
    g_std = np.std(g, ddof=1, axis=0)
    b_std = np.std(b, ddof=1, axis=0)
    print(r_mean)
    print(g_mean)
    print(b_mean)
    print(r_std)
    print(g_std)
    print(b_std)

# print(next(iter(train_dataloaders[0])))
# print(next(iter(test_dataloader)))
# print(next(iter(valid_dataloader)))
# print(len(valid_dataloader))

if __name__ == "__main__":
    pass
