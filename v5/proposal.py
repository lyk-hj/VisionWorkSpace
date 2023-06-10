from anchor import AnchorTarget
from torch import nn
from model_v5 import BackBone
from model_v5 import RPN
from torch.autograd import Variable
import torch
import os
from config import device, cfg
from generate import cal_iou
import numpy as np

rpn_model_path = "../rpn_model/2023_5_23_rpn_1.pt"


class ProposalTarget(nn.Module):
    def __init__(self):
        super(ProposalTarget, self).__init__()
        self.anchor_producer = AnchorTarget()

    def forward(self, image, rpn_model, anchor, sharing=False):
        # image = Variable(image).to(device)
        cla_map, reg_map = rpn_model(image, sharing)
        proposal_boxset = []
        rpn_boxset = []
        for i in range(cla_map.shape[0]):
            for j in range(cla_map.shape[2]):
                for k in range(cla_map.shape[3]):
                    for p in range(cfg.RPN.anchors):
                        cla_vector = torch.softmax(cla_map[i][p * 2: p * 2 + 2, j, k], dim=0)
                        # from IPython import embed;embed()
                        if cla_vector[1] > 0.7:
                            # print(cla_vector[1])
                            correspoding_anchor = anchor[(k * cla_map.shape[3] + k) * p]
                            a_cx = correspoding_anchor[2]
                            a_cy = correspoding_anchor[3]
                            a_w = correspoding_anchor[4]
                            a_h = correspoding_anchor[5]
                            # print(a_cx,a_cy,a_w,a_h)
                            output_box = reg_map[i][p * 4: p * 4 + 4, j, k].to("cpu")
                            output_bbox = np.array(embedding(output_box, [a_cx, a_cy, a_w, a_h]))
                            proposal_box = data2box(output_bbox)
                            if proposal_box[0] < 0 or proposal_box[1] < 0 or proposal_box[2] > cfg.anchor.src_info \
                                    or proposal_box[3] > cfg.anchor.src_info:
                                continue
                            proposal_boxset.append(proposal_box)
                            rpn_boxset.append(output_bbox / cfg.anchor.src_info)
        return proposal_boxset, rpn_boxset


def NMS(predict_proposal):
    # box information:[ltx, lty, rbx, rby, category, confidence]
    effective_boxes = []
    for pp_box in predict_proposal:
        if pp_box[4] != cfg.Head.classes and pp_box[5] > 0.97:
            effective_boxes.append(pp_box)

    effective_boxes = sorted(effective_boxes, key=lambda box: box[5], reverse=True)  # descend
    for i, ei_box in enumerate(effective_boxes):
        for j, ej_box in enumerate(effective_boxes[i + 1:]):
            if ej_box[4] == ei_box[4]:
                iou = cal_iou(ei_box, ej_box)
                if iou > 0.2:
                    effective_boxes[j + i + 1][4] = cfg.Head.classes
    # print(effective_boxes)
    final_eboxes = []
    for eboxes in effective_boxes:
        if eboxes[4] != cfg.Head.classes:
            final_eboxes.append(eboxes)
    return final_eboxes


def data2box(input_bdata):
    ltx = input_bdata[0] - input_bdata[2] / 2
    lty = input_bdata[1] - input_bdata[3] / 2
    rbx = input_bdata[0] + input_bdata[2] / 2
    rby = input_bdata[1] + input_bdata[3] / 2

    return np.array([ltx, lty, rbx, rby])


def embedding(meta, reference):
    cx = float(meta[0]*reference[2] + reference[0])*cfg.anchor.src_info
    cy = float(meta[1]*reference[3] + reference[1])*cfg.anchor.src_info
    w = (np.exp(float(meta[2])) * reference[2])*cfg.anchor.src_info
    h = (np.exp(float(meta[3])) * reference[3])*cfg.anchor.src_info

    return [cx, cy, w, h]


if __name__ == "__main__":
    point_one = [20, 30, 50, 60]
    point_two = [10, 20, 80, 70]
    iou = cal_iou(point_one, point_two)
    print(iou)
