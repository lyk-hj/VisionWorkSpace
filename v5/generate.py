import glob
import random
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import cv2
import torch.nn.functional as F
from config import cfg
from anchor import AnchorTarget, show_anchor

# python rules that height before width, while in cv2.resize is width before height
# input_h = 24
# input_w = 24

BATCH_SIZE = 64
F_BATCH_SIZE = 16

transform = transforms.Compose([
    transforms.RandomRotation(degrees=15, expand=True),
    # transforms.Resize((24, 24)),  # 这里的resize不是裁剪，是缩放
    transforms.ToTensor(),
])

fruit_labels = [l.replace("../fruit_360/Training", "").replace('\\', "") for l in glob.glob(r"../fruit_360/Training/*")]


# print(len(fruit_labels))

# print(fru+it_labels)


def resize(**kwargs):
    data = kwargs.get('data')
    out_size = kwargs.get('out_size')
    return cv2.resize(data, (int(out_size[0]), int(out_size[1])))


# fill not only min edge but also max size
def stretch(**kwargs):
    data = kwargs.get('data')
    out_size = kwargs.get('out_size')
    back = np.zeros(out_size, np.uint8)
    rows, cols = data.shape[0], data.shape[1]
    max_dim, min_dim = (rows, cols) if rows > cols else (cols, rows)
    data = resize(data=data, out_size=(out_size[0] * (cols / max_dim), out_size[0] * (rows / max_dim)))
    h_size_offset = (out_size[0] - data.shape[0]) // 2
    w_size_offset = (out_size[1] - data.shape[1]) // 2
    crop_h_start = h_size_offset
    crop_w_start = w_size_offset
    back[crop_h_start:crop_h_start + data.shape[0], crop_w_start:crop_w_start + data.shape[1]] = data
    return back


# fill the min border
def out_of_shape(**kwargs):
    data = kwargs.get('data')
    out_size = kwargs.get('out_size')
    act_size = kwargs.get('act_size')
    rows, cols = act_size[1], act_size[0]
    max_dim, min_dim = (rows, cols) if rows > cols else (cols, rows)
    data = resize(data=data, out_size=(out_size[0] * (act_size[0] / max_dim), out_size[0] * (act_size[1] / max_dim)))
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


def add_white_boundary(**kwargs):
    data = kwargs.get('data')
    out_size = kwargs.get('out_size')
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
    return resize(data=data, out_size=out_size)


def increase_noise(**kwargs):
    data = kwargs.get('data')
    out_size = kwargs.get('out_size')
    height = data.shape[0]
    weight = data.shape[1]

    for i in range(height):
        for j in range(weight):
            num = random.randint(0, 1000)
            if (num > 996):
                data[i, j] = [0, 0, 0]
            elif (num < 4):
                data[i, j] = [128, 128, 128]
    return resize(data=data, out_size=out_size)


def clip_superfluous_pixel(**kwargs):
    data = kwargs.get('data')
    out_size = kwargs.get('out_size')
    label = kwargs.get('label')
    if label == 0 or label == 1:
        return resize(data=data, out_size=out_size)
    _binary = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    _, _binary = cv2.threshold(_binary, 0, 255, cv2.THRESH_BINARY or cv2.THRESH_OTSU)
    # cv2.imshow("_bin", _binary)
    rows, cols = data.shape[:2]
    row_start = 0
    row_end = rows
    col_start = 0
    col_end = cols
    boundary_flag = 0
    for i in range(rows):
        for j in range(cols):
            if boundary_flag == 0:
                if _binary[i][j] == 255:
                    row_start = i
                    boundary_flag = 1
            elif boundary_flag == 1:
                break
        if boundary_flag == 1:
            if np.sum(_binary[i]) < 255 * 3:
                row_end = i
                break
    boundary_flag = 0
    for i in range(cols):
        for j in range(rows):
            if boundary_flag == 0:
                if _binary[j][i] == 255:
                    col_start = i
                    boundary_flag = 1
            elif boundary_flag == 1:
                break
        if boundary_flag == 1:
            if np.sum(_binary[:, i]) < 255 * 3:
                col_end = i
                break
    # print(row_start, row_end, col_start, col_end)
    row_end = row_end if row_end >= data.shape[0] / 2 else rows
    col_end = col_end if col_end >= data.shape[1] / 2 else cols
    # print(row_start, row_end, col_start, col_end)
    return resize(data=data[row_start:row_end, col_start:col_end], out_size=out_size)


class data_consolidate:
    def __init__(self, data_border, crop_range):
        self.data_border = data_border  # data original border
        self.crop_range = crop_range  # crop_range such as 01234...

        self.functions = {
            1: resize,
            2: stretch,
            3: out_of_shape,
            4: clip_superfluous_pixel,
            5: increase_noise,
            6: add_white_boundary
        }

    def crop(self, **kwargs):
        data = kwargs.get('data')
        crop_mode = kwargs.get('crop_mode')
        crop_h_start = random.randint(0, crop_mode)
        crop_w_start = random.randint(0, crop_mode)
        # print(crop_h_start, crop_w_start)
        return data[crop_h_start:crop_h_start + self.data_border,
               crop_w_start:crop_w_start + self.data_border]

    def random_data_enhance(self, data, label, enhance_flag):
        crop_mode = random.randint(0, self.crop_range)
        out_size = (self.data_border + crop_mode, self.data_border + crop_mode, 3)
        # if self.crop_mode < 5:
        #     # out_size = (28, 28)
        #     out_size = (self.data_border + 4, self.data_border + 4)
        # elif self.crop_mode >= 5 and crop_mode < 10:
        #     out_size = (self.data_border + 2, self.data_border + 2)
        # elif self.crop_mode >= 10 and crop_mode < 14:
        #     out_size = (self.data_border + 1, self.data_border + 1)
        # elif self.crop_mode >= 14 and self.crop_mode < 22:
        #     out_size = (self.data_border + 3, self.data_border + 3)
        # elif self.crop_mode == 22:
        #     out_size = (self.data_border, self.data_border)

        if label == 0:
            # stochastic flip
            data = cv2.flip(data, random.randint(-1, 1)) if random.randint(0, 1) else data
            # random rotation
            height, width = data.shape[:2]
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, random.uniform(-90, 90), 1.0)
            data = cv2.warpAffine(data, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if enhance_flag > 0:
            fun = self.functions[enhance_flag]
            act_long_border = np.random.randint(self.data_border + 5,
                                                self.data_border + 15)  # numpy的randint不包含最大值，python自己的randint包含最大值
            act_size = (out_size[0], act_long_border, 3) if random.randint(0, 1) else (act_long_border, out_size[1])
            data = fun(data=data, out_size=out_size,
                       act_size=act_size, label=label, crop_mode=crop_mode)

            return self.crop(data=data, crop_mode=crop_mode)
        else:
            return resize(data=data, out_size=(self.data_border, self.data_border))


# 通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(data.Dataset):
    # 类初始化
    def __init__(self, root, transform, is_verification):
        self.imgs_path = root
        # self.crop_flag = crop_flag
        self.is_verification = is_verification
        self.transforms = transform
        self.data_border = random.randint(25, 29)
        self.data_enhancer = data_consolidate(self.data_border, crop_range=4)
        # self.data_border = 24
        # print(self.data_border, end="\t")

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
            # crop_mode = self.crop_flag[index]
            enhance_flag = random.randint(-1, 5)
            # print(enhance_flag)
            # data = random_data_enhance(data, label, self.data_border, enhance_flag, crop_mode)
            data = self.data_enhancer.random_data_enhance(data, label, enhance_flag)

        else:
            data = cv2.resize(data, (self.data_border, self.data_border))
        if not 'datac' in img_path:
            # mask1 = np.zeros(data.shape, dtype=np.uint8)
            # mask1 = mask1 + 8
            # mask2 = np.zeros(data.shape, dtype=np.uint8)
            # mask2 = mask2 + 3
            # data = cv2.divide(data, mask1)
            # data = cv2.add(data, mask2)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
            light = random.randint(1, 10)
            data[..., 2] = data[..., 2] / light
            # print(light)
            data = cv2.cvtColor(data, cv2.COLOR_HSV2BGR)
        else:
            pass
            # print("c")
        # cv2.imwrite("sample.jpg", data)
        # cv2.imshow("data", data)
        # cv2.waitKey(0)
        # data = data.astype(np.float32) / 255.0
        data = (data.astype(np.float32) / 255.0 - cfg.generate.mean) / cfg.generate.std
        data = (np.reshape(data.astype(np.float32), (3, data.shape[0], data.shape[1])))
        # print(data.shape)
        return data, label


class FruitDataset(data.Dataset):
    def __init__(self, path, transform, set_mode):
        super(FruitDataset, self).__init__()
        self.imgs_path = path
        self.transforms = transform
        self.set_mode = set_mode
        # self.label = labels

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, item):
        img_path = self.imgs_path[item]
        # print(self.set_mode)
        label = img_path.replace("../fruit_360/" + self.set_mode + "\\", "")
        # print(label)
        label = label[0:label.index("\\")]
        # print(label)
        img = cv2.resize(cv2.imread(img_path), (160, 160))
        label = fruit_labels.index(label)
        # img = img.astype(np.float32) / 255.0
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        img = (img.astype(np.float32) / 255.0 - cfg.fgenerate.mean) / cfg.fgenerate.std
        img = (np.reshape(img.astype(np.float32), (3, img.shape[0], img.shape[1])))
        return img, label


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
data2_path.extend(glob.glob(r"../data2/1*"))
data2_path.extend(glob.glob(r"../data2/5*"))
data3_path = glob.glob(r'../data3/*g')
data7_path = glob.glob(r"../data7/*g")
datac_path = glob.glob(r"../datac/*g")
# print(datac_path)

valid_imgs_path.extend(data0_path)
valid_imgs_path.extend(data3_path)
valid_imgs_path.extend(data7_path)
# valid_imgs_path.extend(datac_path)
# print(valid_imgs_path)

all_imgs_path.extend(data1_path)
all_imgs_path.extend(data2_path)
# all_imgs_path.extend(data7_path)
# all_imgs_path.extend(data3_path)
all_imgs_path.extend(datac_path)
# all_imgs_path.extend(data0_path)

index = np.random.permutation(len(all_imgs_path))
all_imgs_path = np.array(all_imgs_path)[index]  # 打乱数据

a = int(len(all_imgs_path) * 0.8)
# print(a)
train_imgs = all_imgs_path[:a]
test_imgs = all_imgs_path[a:]

train_datasets = [Mydataset(train_imgs, transform, is_verification=False) for _ in
                  range(10)]  # get train dataset rule
train_dataloaders = [data.DataLoader(train_dataset, BATCH_SIZE, True, drop_last=True) for train_dataset in
                     train_datasets]  # get train dataloader

test_dataset = Mydataset(test_imgs, transform, is_verification=False)  # get test dataset rule
test_dataloader = data.DataLoader(test_dataset, BATCH_SIZE, True, drop_last=True)  # get test dataloader

valid_dataset = Mydataset(valid_imgs_path, transform, is_verification=True)
valid_dataloader = data.DataLoader(valid_dataset, BATCH_SIZE, True, drop_last=True)

fruit_train = glob.glob(r"../fruit_360/Training/*")
fruit_test = glob.glob(r"../fruit_360/Test/*")
fruit_valid = glob.glob(r"../fruit_360/Validation/*")

# print(fruit_train)
fruit_train_data = []
for p in fruit_train:
    fruit_train_data.extend(glob.glob(p + "/*"))
# print(fruit_train_data)

fruit_test_data = []
for p in fruit_test:
    fruit_test_data.extend(glob.glob(p + "/*"))

fruit_valid_data = []
for p in fruit_valid:
    fruit_valid_data.extend(glob.glob(p + "/*"))

f_train_datasets = [FruitDataset(fruit_train_data, transform, set_mode="Training") for _ in range(10)]
f_train_dataloaders = [data.DataLoader(f_train_dataset, F_BATCH_SIZE, shuffle=True, drop_last=True) for f_train_dataset
                       in f_train_datasets]

f_test_dataset = FruitDataset(fruit_test_data, transform, set_mode="Test")
f_test_dataloader = data.DataLoader(f_test_dataset, F_BATCH_SIZE, shuffle=True, drop_last=False)

f_valid_dataset = FruitDataset(fruit_valid_data, transform, set_mode="Validation")
f_valid_dataloader = data.DataLoader(f_valid_dataset, F_BATCH_SIZE, shuffle=True, drop_last=False)


def get_mean_std(dataset):
    r = []
    g = []
    b = []
    for i, j in dataset:
        r.append(i[0])
        g.append(i[1])
        b.append(i[2])
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    r_std = np.std(r, ddof=1)
    g_std = np.std(g, ddof=1)
    b_std = np.std(b, ddof=1)
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
# print(next(iter(f_train_dataloaders[0])))
# print(next(iter(f_test_dataloader)))
# print(next(iter(f_valid_dataloader)))
# print(len(f_train_dataloaders[0]))


def rpn_label(exhibit_anchor=False, log=False):
    img_files = glob.glob("../demo/datasets/images/*")
    label_files = glob.glob("../demo/datasets/txt/*")
    anchor_producer = AnchorTarget()
    coarse_anchors, anchors_index = anchor_producer((cfg.anchor.feature_map_size, cfg.anchor.feature_map_size))
    if exhibit_anchor:
        show_anchor(coarse_anchors[anchors_index, :])

    final_label_datas = []
    final_default_datas = []
    for file in label_files:
        # get anchor for this image
        default_boxes, default_datas = transform_default_boxes(coarse_anchors)
        # print(len(default_boxes))
        # print(len(default_datas))

        # get label box for this image
        f_txt = open(file)
        label_boxes, label_datas = transform_label_boxes(f_txt)
        img_index = file.split('/')[-1].split('\\')[-1].split('.')[0]
        picture = cv2.imread("../demo/datasets/images/"+img_index+".jpg")
        # show_anchor(np.array(label_boxes)*cfg.anchor.src_info, picture)
        # print(len(label_boxes))
        # print(len(label_datas))
        # print("---------------------------------------------------------------")

        # label the anchors
        positive_anchor = []
        appearance_count = np.zeros(cfg.Head.classes)
        for j, label_box in enumerate(label_boxes):
            positive_sanchor = []
            positive_sanchor_index = []
            for i in anchors_index:
                iou = cal_iou(default_boxes[i], label_box)
                if iou > default_datas[i][0]:
                    default_datas[i][0] = iou
                    if iou > 0.2:
                        default_datas[i][1] = label_datas[j][0]
                        positive_sanchor.append(coarse_anchors[i])
                        positive_sanchor_index.append(i)
                        default_datas[i][6] = appearance_count[int(label_datas[j][0])]
                        # print(default_datas[i][6])
                        # print(iou)
                    elif iou < 0.1:
                        default_datas[i][1] = cfg.Head.classes  # background
            appearance_count[int(label_datas[j][0])] += 1
            filter_count = 150
            if len(positive_sanchor) > filter_count:
                positive_sanchor, positive_sanchor_index, default_datas = \
                    filtrate_positive(positive_sanchor, positive_sanchor_index, default_datas, filter_count)
            positive_anchor.extend(positive_sanchor)

        # show_anchor(positive_anchor)
        # print(len(positive_anchor))
        # print(default_datas)
        final_label_datas.append(label_datas)
        final_default_datas.append(default_datas)
    # print(len(final_label_datas[0]))
    # print(len(final_default_datas[0]))

    # deal with image data
    image_datas = []
    images_origin = []
    for img in img_files:
        src = cv2.imread(img)
        images_origin.append(src)
        # print(src.shape)
        src = np.reshape(src, (1, 3, 224, 224))
        src = src.astype(np.float32) / 255.0
        src = torch.tensor(src)
        image_datas.append(src)

    # print(image_datas)

    return final_label_datas, final_default_datas, image_datas, images_origin


def cal_iou(anchor, label_box):
    # box should be [left_topx, left_topy, right_bottomx, right_bottomy](normed)
    ins_left_topx = max(anchor[0], label_box[0]) if label_box[0] < anchor[0] < label_box[2] \
                                                    or anchor[0] < label_box[0] < anchor[2] else None
    ins_left_topy = max(anchor[1], label_box[1]) if label_box[1] < anchor[1] < label_box[3] \
                                                    or anchor[1] < label_box[1] < anchor[3] else None
    if ins_left_topx is None or ins_left_topy is None:
        return 0
    ins_right_bottomx = min(anchor[2], label_box[2])
    ins_right_bottomy = min(anchor[3], label_box[3])
    intersection_area = (ins_right_bottomx - ins_left_topx) * (ins_right_bottomy - ins_left_topy)
    combination_area = (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]) + (label_box[2] - label_box[0]) * (
            label_box[3] - label_box[1]) - intersection_area
    iou = intersection_area / combination_area
    return iou if iou <= 1 else 1


def transform_label_boxes(f_txt):
    label_datas = []
    for line in f_txt.readlines():
        single_data = [float(text) for text in line.split(' ')]
        label_datas.append(single_data)
        # print(line)
    # print(label_datas)
    label_boxes = [[ldata[1] - ldata[3] / 2.0, ldata[2] - ldata[4] / 2.0,
                    ldata[1] + ldata[3] / 2.0, ldata[2] + ldata[4] / 2.0] for ldata in label_datas]

    return label_boxes, label_datas


def transform_default_boxes(anchors):
    default_boxes = anchors / cfg.anchor.src_info
    # first zero for iou and second zero for category
    default_datas = [[0, -1,
                      (anchor[0] + anchor[2]) / (2*cfg.anchor.src_info),
                      (anchor[1] + anchor[3]) / (2*cfg.anchor.src_info),
                      (anchor[2] - anchor[0]) / cfg.anchor.src_info,
                      (anchor[3] - anchor[1]) / cfg.anchor.src_info,
                      -1] for anchor in anchors]
    default_datas = np.array(default_datas)
    return default_boxes, default_datas


def filtrate_positive(positive_sanchor, positive_sanchor_index, default_datas, filter_count):
    zipped_positive = zip(positive_sanchor, positive_sanchor_index)
    zipped_positive = sorted(zipped_positive, key=lambda x: default_datas[x[1]][0], reverse=True)
    sorted_positive = zip(*zipped_positive)
    # from IPython import embed; embed()
    positive_sanchor, positive_sanchor_index = [list(x) for x in sorted_positive]
    for i in range(filter_count, len(positive_sanchor)):
        default_datas[positive_sanchor_index[i]][1] = -1
        # print(default_datas[positive_sanchor_index[i]])
    # for i in range(200):
    #     print(default_datas[positive_sanchor_index[i]][0])
    positive_sanchor = positive_sanchor[:filter_count]
    positive_sanchor_index = positive_sanchor_index[:filter_count]
    # print(default_datas)
    return positive_sanchor, positive_sanchor_index, default_datas

if __name__ == "__main__":
    # get_mean_std(f_train_datasets[0])
    rpn_label(exhibit_anchor=True, log=True)
