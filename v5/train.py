# from visualize_train import hl_visualize
import copy
import os

import hiddenlayer as hl
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import AdamW

import generate as hj_generate
from anchor import show_anchor
from config import device, cfg
from generate import rpn_label
from loss import RpnLoss, DetectLoss
from model_v5 import ClsModel
from model_v5 import RPN, Head
from proposal import ProposalTarget

file_name = '2023_4_23_fruit_1'
new_name = '2023_4_23_fruit_1'
pre_train_name = '2023_3_11_hj_num_1'
model_path = '../weight/' + file_name + '.pt'
save_path = '../weight/' + new_name + '.pt'
pre_train_path = '../weight/' + pre_train_name + '.pt'

rpn_save_path = "../weight/rpn_2023_6_9_1.pt"
detect_save_path = "../weight/rec_2023_6_9_1.pt"
backbone_save_path = "../weight/bcb_2023_6_9_1.pt"

rpn_pre_path = "../weight/rpn_2023_6_9_1.pt"
detect_pre_path = "../weight/rec_2023_6_9_1.pt"
backbone_pre_path = "../weight/bcb_2023_6_9_1.pt"


print(device)


def show(result):
    print("Loss                     || {}\n"
          "--------------------------confidence------\t\t------equality------\n"
          "Train Accuracy       || {}\t\t{}\n"
          "Test Accuracy        || {}\t\t{}\n"
          "Valid Accuracy       || {}\t\t{}\n".format
          (result[0],
           result[1], result[2],
           result[3], result[4],
           result[5], result[6]))


def normalized_train(data):
    return data / (len(hj_generate.f_train_dataloaders[0]) * 10)


def normalized_test(data):
    return data / len(hj_generate.f_test_dataloader)


def normalized_valid(data):
    return data / len(hj_generate.f_valid_dataloader)


def fine_turning_model(model):
    # model.dense = ClsModel().dense
    for k, i in model.named_parameters():
        # if ("conv4" not in k) and ('dense' not in k):
        i.requires_grad = False
    # for i in model.parameters():
    #     print(i.requires_grad)
    return model


def freezing_model(model):
    for k, i in model.named_parameters():
        i.requires_grad = False
    return model


def infer_data_process(model, data_loader):
    infer_correct = 0
    _infer_correct = 0
    for data, label in data_loader:
        x_infer, y_infer = data, label
        x_infer, y_infer = Variable(x_infer).to(device), Variable(y_infer).to(device)

        # obtain output
        outputs = model(x_infer)
        value, pred = torch.max(outputs.data, 1)

        # calculate correct value
        value = value * (pred == y_infer.data)

        infer_correct += torch.sum(value) / label.shape[0]
        _infer_correct += torch.sum(pred == y_infer.data) / label.shape[0]
    return float(infer_correct) / len(data_loader), float(_infer_correct) / len(data_loader)


def export_train_data(train_result):
    # 1 train_loss train_confidence train_equality test_confidence test_equality valid_confidence valid_equality\n
    train_output = "../weight/" + file_name + '/'
    if not os.path.exists(train_output):
        os.mkdir(train_output)
    with open(train_output + 'train_result.txt', 'a') as f_txt:
        with open(train_output + 'train_result.txt', 'r') as n_txt:
            memories = int(len(n_txt.readlines()) / 2)
        n_txt.close()
        content = str(memories + 1)
        content += "\t\t  loss_val"
        content += "\t\t  train_c"
        content += "\t\t  train_e"
        content += "\t\t  test_c"
        content += "\t\t  test_e"
        content += "\t\t  valid_c"
        content += "\t\t  valid_e"
        content += "\n"
        for result in train_result:
            content += ("\t\t  " + str("{:5f}".format(result)))
        content += '\n'
        f_txt.write(content)
    f_txt.close()


def cls_train():
    # load saved model
    # model = torch.load(model_path, map_location=device)

    # fine_turning model from pre train model
    # model = fine_turning_model(torch.load(pre_train_path, map_location=device))

    # train a new model
    model = ClsModel()

    # model.load_state_dict(torch.load(model_path))
    epochs = 20
    lr = 0.001
    log_step = 100
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.005)
    # optimizer=SGD(model.parameters(),lr=lr,weight_decay=0.09,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print(model)
    model = model.to(device)

    best_model = copy.deepcopy(model)
    best_accuracy = 0.5
    history = hl.History()
    canvas = hl.Canvas()
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch + 1, epochs))
        print('-' * 10)
        running_loss = 0.0
        running_correct = 0
        _running_correct = 0
        batch_step = 1
        model.train()

        for f_train_dataloader in hj_generate.f_train_dataloaders:
            for step, (data, label) in enumerate(f_train_dataloader):
                batch_step += 1
                X_train, y_train = data, label
                X_train, y_train = Variable(X_train).to(device), Variable(y_train).to(device)
                outputs = model(X_train)
                # print(outputs.shape)
                value, pred = torch.max(outputs.data, 1)

                # loss backward
                optimizer.zero_grad()
                loss = loss_fn(outputs, y_train.long())  # already divide the shape 0 of labels
                loss.backward()
                optimizer.step()

                # calculate result
                value = value * (pred == y_train.data)
                running_loss += loss.data
                running_correct += torch.sum(value) / label.shape[0]
                _running_correct += torch.sum(pred == y_train.data) / label.shape[0]
                if batch_step % log_step == 0:
                    print(data.shape)
                    # training loss
                    training_pre_loss = running_loss / batch_step

                    # distinguish accuracy(confidence)
                    training_pre_correct = running_correct / batch_step

                    # distinguish accuracy(equality)
                    _training_pre_correct = _running_correct / batch_step

                    # valid_pre_correct, _valid_pre_correct \
                    #     = infer_data_process(model, hj_generate.valid_dataloader)

                    print("[loss]:{}\t[tcc]:{}\t[tce]:{}".format(training_pre_loss,
                                                                 training_pre_correct,
                                                                 _training_pre_correct))
                    history.log((epoch, batch_step),
                                train_loss=training_pre_loss,
                                train_acc=training_pre_correct,
                                layer_weight=torch.reshape(model.dense[1].conv.weight, torch.Size([32, 256])))

                    with canvas:
                        canvas.draw_plot(history["train_loss"])
                        canvas.draw_plot(history["train_acc"])
                        canvas.draw_image(history["layer_weight"], )
                    print('-' * 10)
                    # print(torch.reshape(model.dense[1].conv.weight, torch.Size([32, 256])))

        model.eval()
        # implement in test set
        testing_pre_correct, _testing_pre_correct \
            = infer_data_process(model, hj_generate.f_test_dataloader)

        # implement in valid set
        valid_pre_correct, _valid_pre_correct \
            = infer_data_process(model, hj_generate.f_valid_dataloader)

        # total average loss
        running_pre_loss = normalized_train(running_loss)

        # total average confidence correct
        running_pre_correct = normalized_train(running_correct)

        # total average equality correct
        _running_pre_correct = normalized_train(_running_correct)

        if 0.2 * running_pre_correct + 0.3 * testing_pre_correct + 0.5 * valid_pre_correct > best_accuracy:
            print("more excellent!!!")
            best_accuracy = 0.2 * running_pre_correct + 0.3 * testing_pre_correct + 0.5 * valid_pre_correct
            best_model = copy.deepcopy(model)

        result = [running_pre_loss,
                  running_pre_correct, _running_pre_correct,
                  testing_pre_correct, _testing_pre_correct,
                  valid_pre_correct, _valid_pre_correct]

        show(result)
        export_train_data(result)

        scheduler.step()
    torch.save(best_model, save_path)
    # torch.onnx.export(best_model, input_data, save_path, opset_version=9, verbose=True, input_names=input_names,
    #                   output_names=output_names, dynamic_axes={'input': {0: '1'}}, )#export model direct


def rpn_train(pre_train_model=None, backbone=None):
    rpn = RPN().to(device)
    if isinstance(pre_train_model, torch.nn.Module):
        rpn = pre_train_model.to(device)
    if isinstance(backbone, torch.nn.Module):
        backbone = backbone.to(device)
        rpn.feature_extractor = freezing_model(backbone)
    # for k, i in rpn.feature_extractor.named_parameters():
    #     print(i.requires_grad)

    epochs = 45
    lr = 0.01
    optimizer = Adam(rpn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss_fn = RpnLoss()

    for epoch in range(epochs):
        train_loss = 0
        label_datas, anchor_datas, image_datas, _ = rpn_label()
        # from IPython import embed;embed()
        for label_data, anchor_data, image_data in zip(label_datas, anchor_datas, image_datas):
            image_data = Variable(image_data).to(device)
            # print(image_data.requires_grad)
            rpn_class_map, rpn_regress_map = rpn(image_data)
            # loss_backward
            optimizer.zero_grad()
            loss = loss_fn(rpn_class_map, rpn_regress_map, label_data, anchor_data)
            loss.backward()
            # print(rpn.conv1.conv.weight.grad)
            train_loss += float(loss.data)
            optimizer.step()
        print("[Train_Loss]        || {}".format(train_loss / len(label_datas)))
        scheduler.step()
    torch.save(rpn, rpn_save_path)


def detector_train(detect_model=None, rpn_model=None, backbone=None, sharing=False):
    detector = Head().to(device)
    if isinstance(detect_model, torch.nn.Module):
        detector = detect_model.to(device)
        if sharing and isinstance(backbone, torch.nn.Module):
            backbone = freezing_model(backbone)
            detector.feature_extractor = freezing_model(backbone)
    # print(detector)
    # for k, i in detector.feature_extractor.named_parameters():
    #     print(i.requires_grad)

    proponent = ProposalTarget().to("cpu").eval()

    if rpn_model is None:
        print("[ERROR!] have not given rpn model!!!!")
    rpn = rpn_model.to("cpu").eval()

    epochs = 30
    lr = 0.001
    optimizer = Adam(detector.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = DetectLoss()

    for epoch in range(epochs):
        train_loss = 0
        label_datas, anchor_datas, image_datas, image_origin = rpn_label()
        for label_data, anchor_data, image, src in zip(label_datas, anchor_datas, image_datas, image_origin):
            # proposal
            with torch.no_grad():
                if backbone is None:
                    proposal_set, rpn_boxset = proponent(image, rpn, anchor_data, sharing)
                else:
                    feature_map = backbone(image)
                    proposal_set, rpn_boxset = proponent(feature_map, rpn, anchor_data, sharing)
            # for k, i in rpn.feature_extractor.named_parameters():
            #     print(i.requires_grad)
            # show_anchor(proposal_set, src)

            # input proposal and feature map to predict
            train_step_loss = 0
            train_step = 0
            x = image.to(device) if backbone is None else feature_map.to(device)
            for j in range(0, len(proposal_set), cfg.Head.batch_size):
                # print("-----------------------------")
                # for k, i in detector.named_parameters():
                #     print(i.requires_grad)

                if (len(proposal_set) - j) // cfg.Head.batch_size == 0:
                    proposal_batch = proposal_set[j:]
                    rpn_batch = rpn_boxset[j:]
                else:
                    proposal_batch = proposal_set[j: j + cfg.Head.batch_size]
                    rpn_batch = rpn_boxset[j: j + cfg.Head.batch_size]

                if len(proposal_batch) == 1:
                    continue
                output = detector(x, proposal_batch, sharing)
                # print(image.requires_grad)
                optimizer.zero_grad()
                loss = loss_fn(label_data, proposal_batch, rpn_batch, output, src)
                loss.backward()
                # print(detector.dense[2].fc.weight.grad[0][0])
                optimizer.step()
                train_step += 1
                train_step_loss += float(loss.item())  # after item is an element not a tensor, can't be trained
                del loss
                del output
            train_loss += (train_step_loss / train_step)
            print("[Train_step_Loss]        || {}".format(float(train_step_loss / train_step)))
        train_loss /= len(image_datas)
        print("[Train_Loss]        || {}".format(float(train_loss)))
        scheduler.step()
    torch.save(detector, detect_save_path)
    torch.save(detector.feature_extractor, backbone_save_path)


def alternative_train():
    # first step
    print("[first step ] ------------------------------------------")
    rpn_train()
    # second step
    print("[second step] ------------------------------------------")
    rpn_model = torch.load(rpn_pre_path)
    detector_train(detect_model=None, rpn_model=rpn_model, backbone=None, sharing=False)
    # third step
    print("[third step ] ------------------------------------------")
    detect_model = torch.load(detect_pre_path)
    backbone_model = torch.load(backbone_pre_path)
    rpn_train(pre_train_model=rpn_model, backbone=backbone_model)
    # forth step
    print("[forth step ] ------------------------------------------")
    # rpn_model = torch.load(rpn_pre_path)
    detector_train(detect_model=detect_model, rpn_model=rpn_model, backbone=backbone_model, sharing=True)



if __name__ == "__main__":
    # cls_train()
    # rpn_train(rpn_model, detect_model.feature_extractor)
    alternative_train()



