import copy

import cv2
import numpy as np
import random
import torch
import copy
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import SmoothL1Loss

from config import device, cfg
from detect import embedding
from generate import cal_iou


def limit(number):
    if number > cfg.anchor.src_info:
        return cfg.anchor.src_info
    elif number < 0:
        return 0
    else:
        return number


def detect_validation(predict_regression, proposal_box, predict_cla, pic):
    pic_temp = copy.deepcopy(pic)
    predict_reg = predict_regression.cpu().detach().numpy()
    detect_box = embedding(predict_reg.data, proposal_box)
    predict_cla = torch.softmax(torch.reshape(predict_cla.data, (1, 3)), dim=1)
    _, detect_class = torch.max(predict_cla, dim=1)
    if detect_class == cfg.Head.classes:
        return
    pt1 = (int(limit(detect_box[0] - detect_box[2] / 2)), int(limit(detect_box[1] - detect_box[3] / 2)))
    pt2 = (int(limit(detect_box[0] + detect_box[2] / 2)), int(limit(detect_box[3] + detect_box[3] / 2)))
    cv2.putText(pic_temp, cfg.Head.class_names[detect_class], pt1, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    cv2.rectangle(pic_temp, pt1, pt2, (0, 0, 255))
    name = "./validation/" + str(random.randint(0, 100)) + ".jpg"
    cv2.imwrite(name, pic_temp)


class RpnLoss(nn.Module):
    def __init__(self, parameter=None):
        super(RpnLoss, self).__init__()
        self.parameter = parameter
        self.loss_regression = SmoothL1Loss()
        # cross entry for classification
        self.loss_classification = CrossEntropyLoss()

    def forward(self, cla_map, reg_map, label_data, anchor_data):
        # from IPython import embed; embed()
        tt_loss = torch.zeros(1, device=device)
        positive_count = []
        # positive anchor
        for j in range(cla_map.shape[0]):
            positive_count_temp = 0
            for i in range(len(anchor_data)):
                if anchor_data[i][1] != -1 and anchor_data[i][1] != cfg.Head.classes:
                    positive_count_temp += 1
                    # anchor size data
                    a_cx = anchor_data[i][2]
                    a_cy = anchor_data[i][3]
                    a_w = anchor_data[i][4]
                    a_h = anchor_data[i][5]
                    # anchor category data
                    a_cls = anchor_data[i][1]

                    # predict size data(relative location)
                    reflect_channel = (i % cfg.RPN.anchors) * 4
                    reflect_position_h = i // cfg.RPN.anchors % cfg.anchor.feature_map_size
                    reflect_position_w = i // cfg.RPN.anchors // cfg.anchor.feature_map_size
                    # from IPython import embed;embed()
                    p_cx = reg_map[j][reflect_channel][reflect_position_h][reflect_position_w]  # map is CHW
                    p_cy = reg_map[j][reflect_channel + 1][reflect_position_h][reflect_position_w]
                    p_w = reg_map[j][reflect_channel + 2][reflect_position_h][reflect_position_w]
                    p_h = reg_map[j][reflect_channel + 3][reflect_position_h][reflect_position_w]
                    # predict category data
                    reflect_channel = (i % cfg.RPN.anchors) * 2
                    # from IPython import embed;embed()
                    # map is CHW
                    p_cls = cla_map[j][reflect_channel: reflect_channel + 2, reflect_position_h, reflect_position_w]

                    count = 0
                    g_cx = None
                    g_cy = None
                    g_w = None
                    g_h = None
                    g_cls = None
                    for label in label_data:
                        if label[0] == anchor_data[i][1]:
                            if count == anchor_data[i][6]:
                                # ground truth size data
                                g_cx = label[1]
                                g_cy = label[2]
                                g_w = label[3]
                                g_h = label[4]
                                # ground truth category data
                                g_cls = int(label[0])
                            else:
                                count += 1

                    # cal the relative loc of anchor
                    t_cx = (g_cx - a_cx) / a_w
                    t_cy = (g_cy - a_cy) / a_h
                    t_w = np.log(g_w / a_w)
                    t_h = np.log(g_h / a_h)

                    # loss for regression
                    t_tensor = torch.Tensor([t_cx, t_cy, t_w, t_h])
                    p_tensor = torch.Tensor([p_cx, p_cy, p_w, p_h])
                    loss_reg = self.loss_regression(p_tensor, t_tensor)

                    # loss for classification
                    tc_tensor = torch.zeros(2, device=device)
                    tc_tensor[1] = 1
                    pc_tensor = torch.Tensor(p_cls)
                    loss_cls = self.loss_classification(pc_tensor, tc_tensor)

                    # sum loss_reg and loss_cls
                    tt_loss += (loss_cls + 4 * loss_reg)  # 4 is for balance, negative no regression
            positive_count.append(positive_count_temp)
        # negative anchor
        negative_count = []
        for j in range(cla_map.shape[0]):
            negative_count_temp = 0
            for i in random.sample(range(len(anchor_data)), 300):
                if anchor_data[i][1] == cfg.Head.classes:
                    # restrict n:p=3:1
                    if negative_count_temp >= positive_count[j] * 3:
                        break
                    negative_count_temp += 1
                    # predict class data
                    reflect_channel = (i % cfg.RPN.anchors) * 2
                    reflect_position_h = i // cfg.RPN.anchors % cfg.anchor.feature_map_size
                    reflect_position_w = i // cfg.RPN.anchors // cfg.anchor.feature_map_size
                    p_cls = cla_map[j][reflect_channel: reflect_channel + 2, reflect_position_h, reflect_position_w]

                    # classification loss
                    pc_tensor = torch.Tensor(p_cls)
                    tc_tensor = torch.zeros(2, device=device)
                    tc_tensor[0] = 1
                    loss_cls = self.loss_classification(pc_tensor, tc_tensor)

                    # add to tt_loss
                    tt_loss += loss_cls
            negative_count.append(negative_count_temp)
        # from IPython import embed;embed()
        final_loss = tt_loss / (sum(positive_count) + sum(negative_count))
        return final_loss


class DetectLoss(nn.Module):
    def __init__(self):
        super(DetectLoss, self).__init__()
        self.loss_classification = CrossEntropyLoss()
        self.loss_location = SmoothL1Loss()

    def forward(self, label_data, proposal_batch, rpn_batch, predict_data, pic):
        tt_loss = torch.zeros(1, device=device)
        pos_count = 0
        nag_count = 0
        for j, (proposal_box, rpn_data) in enumerate(zip(proposal_batch, rpn_batch)):
            proposal_box = proposal_box / cfg.anchor.src_info
            predict_classification = predict_data[j][:3]
            predict_regression = predict_data[j][3:]
            # from IPython import embed; embed()
            iou_temp = 0
            index = 0
            for i, label in enumerate(label_data):
                label_box = [label[1] - label[3] / 2.0, label[2] - label[4] / 2.0,
                             label[1] + label[3] / 2.0, label[2] + label[4] / 2.0]
                iou = cal_iou(proposal_box, label_box)
                # from IPython import embed;embed()
                if iou > iou_temp:  # should not classification coincident
                    iou_temp = iou
                    index = i
            if iou_temp > 0.3:
                if random.randint(0, 100) < 20:
                    detect_validation(predict_regression=predict_regression, proposal_box=proposal_box,
                                      predict_cla=predict_classification, pic=pic)
                # from IPython import embed;embed()
                # ground truth location embedding
                g_cx = (label_data[index][1] - rpn_data[0]) / rpn_data[2]
                g_cy = (label_data[index][2] - rpn_data[1]) / rpn_data[3]
                g_w = np.log(label_data[index][3] / rpn_data[2])
                g_h = np.log(label_data[index][4] / rpn_data[3])

                # loss for regression
                g_tensor = torch.Tensor([[g_cx, g_cy, g_w, g_h]]).to(device)
                p_tensor = torch.reshape(predict_regression, torch.Size((1, 4)))
                loss_reg = self.loss_location(p_tensor, g_tensor)

                # loss for classification
                label_classification = torch.Tensor([int(label_data[index][0])]).to(device)
                predict_tensor = torch.reshape(predict_classification, torch.Size((1, 3)))
                loss_cla = self.loss_classification(predict_tensor, label_classification.long())

                ratio = 8
                tt_loss += ratio * (loss_reg + loss_cla)
                pos_count += 1
                # print(predict_classification)
            else:
                # loss for classification
                label_classification = torch.Tensor([2]).to(device)
                predict_tensor = torch.reshape(predict_classification, torch.Size((1, 3)))
                loss_cla = self.loss_classification(predict_tensor, label_classification.long())
                tt_loss += loss_cla
                nag_count += 1
        tt_loss /= len(proposal_batch)
        # print(tt_loss)
        return tt_loss


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda_version)
