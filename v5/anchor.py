import numpy as np
from torch import nn
from model_v5 import RPN
from config import cfg
import torch
import cv2


# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=9, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    # print(ratio_anchors)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    # print(anchors)
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = size_anchor(anchor)  # get w, h, center x and center y
    size = w * h
    # all these three is list
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = norm_anchor(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = size_anchor(anchor)
    ws = w * scales
    hs = h * scales
    anchors = norm_anchor(ws, hs, x_ctr, y_ctr)
    return anchors


def size_anchor(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """
    # print(anchor[2])
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def norm_anchor(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    # print(ws[:])
    ws = ws[:, np.newaxis]
    # print(ws)
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))  # anchors is tl xy and rb xy format
    return anchors


class AnchorTarget(nn.Module):
    def __init__(self):
        super(AnchorTarget, self).__init__()
        self.anchors = generate_anchors()
        self.num_anchors = self.anchors.shape[0]

    def forward(self, shape):
        width, height = shape[0], shape[1]
        shift_x = (np.arange(0, width) + 0.5) * (cfg.anchor.src_info // width)
        shift_y = (np.arange(0, height) + 0.5) * (cfg.anchor.src_info // height)
        # shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        # print(shift_x, shift_y)
        # shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
        #                     shift_x.ravel(), shift_y.ravel())).transpose()
        boxes = []
        for sx in shift_x:
            for sy in shift_y:
                # print(sx,sy)
                left_top_x = sx + self.anchors[:, 0]
                # print(left_top_x)
                left_top_y = sy + self.anchors[:, 1]
                # print(left_top_y)
                right_bottom_x = sx + self.anchors[:, 2]
                # print(right_bottom_x)
                right_bottom_y = sy + self.anchors[:, 3]
                # print(right_bottom_y)
                box_set = zip(left_top_x, left_top_y, right_bottom_x, right_bottom_y)
                for t in box_set:
                    boxes.append(list(t))

        boxes = np.reshape(boxes, (-1, 4))
        # print(len(boxes))
        final_boxes_index = np.where(
            (boxes[:, 0] >= -cfg.anchor.allowed_border) &
            (boxes[:, 1] >= -cfg.anchor.allowed_border) &
            (boxes[:, 2] < cfg.anchor.src_info + cfg.anchor.allowed_border) &
            (boxes[:, 3] < cfg.anchor.src_info + cfg.anchor.allowed_border)
        )[0]
        # print(final_boxes_index)

        # final_boxes = boxes[final_boxes_index, :]

        return boxes, final_boxes_index


def show_anchor(boxes, picture=None):
    if picture is None:
        img = np.zeros((1200, 1200, 3), np.uint8)
        start = (1200 - cfg.anchor.src_info) // 2
        end = 1200 - start
        cv2.circle(img, (start, start), 2, (0, 0, 255), -1)
        cv2.circle(img, (start, end), 2, (0, 0, 255), -1)
        cv2.circle(img, (end, end), 2, (0, 0, 255), -1)
        cv2.circle(img, (end, start), 2, (0, 0, 255), -1)
    else:
        img = picture
        start = 0
    for i, box in enumerate(boxes):
        cv2.rectangle(img, (int(box[0]) + start, int(box[1]) + start), (int(box[2]) + start, int(box[3]) + start),
                      (255, 0, 0))
        # if i == 8:
        #     break
        # print(box)
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    import time

    t = time.time()
    # data = torch.randn((1, 3, 60, 40))  # 60*40 still too huge
    data = (60, 40)
    anchorTarget = AnchorTarget()
    boxes, final_boxes_index = anchorTarget(data)
    final_boxes = boxes[final_boxes_index, :]
    print(len(final_boxes))
    show_anchor(final_boxes)
