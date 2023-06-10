import torch
from torch import nn
from torch.autograd import Variable
from proposal import ProposalTarget, NMS, data2box, embedding
from generate import rpn_label
from anchor import show_anchor
import numpy as np
import cv2
from model_v5 import RPN, Head, BackBone
from config import cfg

test_device = "cpu"
rpn_path = "../weight/rpn_2023_6_9_1.pt"
detect_path = "../weight/rec_2023_6_9_1.pt"
backb_path = "../weight/bcb_2023_6_9_1.pt"


def detect(image, ori):
    rpn_model = torch.load(rpn_path).to(test_device)
    detect_model = torch.load(detect_path).to(test_device)
    backbone = torch.load(backb_path).to(test_device)
    producer = ProposalTarget()
    _, anchor_datas, _, ori_pic = rpn_label()
    anchors = anchor_datas[0]
    # print(anchors[0]*cfg.anchor.src_info)

    if not whether_feature_identical(rpn_model, detect_model):
        print("[ERROR!] rpn differs from detect in feature extracting component!!!!")
        return
    image = torch.Tensor(image).to(test_device)

    feature_map = backbone(image)
    proposals, rpns = producer(feature_map, rpn_model, anchors, sharing=True)
    output = detect_model(feature_map, proposals, sharing=True)

    coarse_results = []
    for i in range(output.shape[0]):
        output_cla = torch.reshape(output[i][:3], torch.Size((1, 3)))
        # print(output_cla)
        output_reg = torch.reshape(output[i][3:], torch.Size((1, 4)))
        confidence, index = torch.max(output_cla, 1)
        # print(rpns[i])
        # print(output_reg)
        output_result = embedding(output_reg[0].detach().numpy(), rpns[i])
        output_result = list(data2box(output_result))
        output_result.extend([index.item(), confidence.item()])
        # print(output_result)
        # if output_result[4] != 2:
        #     print(output_result)
        coarse_results.append(output_result)
    print(len(coarse_results))
    final_result = NMS(coarse_results)
    print(len(final_result))
    show_detect(final_result, ori)


def whether_feature_identical(rpn_model, detect_model):
    rpn_structure = []
    detect_structure = []
    for name, module in rpn_model.feature_extractor.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
            rpn_structure.append(module.weight.data)

    # Extract features from net2
    for name, module in detect_model.feature_extractor.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.BatchNorm2d):
            detect_structure.append(module.weight.data)

    # print(len(rpn_structure))
    # print(len(detect_structure))
    for i in range(len(rpn_structure)):
        if not torch.equal(rpn_structure[i], detect_structure[i]):
            print(rpn_structure[i])
            print(detect_structure[i])
            print("diff")
            return False
    return True


def show_detect(results, img):
    for i, box in enumerate(results):
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0))
        cv2.putText(img, cfg.Head.class_names[int(box[4])], (int(box[0]), int(box[1])),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255))
    cv2.imshow("img", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    ori_src = cv2.resize(cv2.imread("../demo/datasets/images/08.jpg"), (224, 224))
    src = np.reshape(ori_src, (1, 3, 224, 224))
    src = src.astype(np.float32) / 255.0
    detect(src, ori_src)

