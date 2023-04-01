import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW
from torch.autograd import Variable
from model_v4 import MultiTaskModel
import hj_generate_v4_5 as hj_generate
import torch.onnx
import os
import cv2

file_name = '2023_4_1_hj_num_1'
new_name = '2023_3_27_hj_num_1'
pre_train_name = '2023_3_11_hj_num_1'
model_path = '../weight/' + file_name + '.pt'
save_path = '../weight/' + new_name + '.pt'
pre_train_path = '../weight/' + pre_train_name + '.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# train_loss,
# train_acc_obj, train_acc_cla, _train_acc_obj, _train_acc_cla,
# test_acc_obj, test_acc_cla, _test_acc_obj, _test_acc_cla,
# valid_acc_obj, valid_acc_cla, _valid_acc_obj, _valid_acc_cla,
def show(result):
    print("Loss                     || {}\n"
          "------------------------------confidence------\t\t------equality------\n"
          "Train obj Accuracy       || {}\t\t{}\n"
          "Test obj Accuracy        || {}\t\t{}\n"
          "Valid obj Accuracy       || {}\t\t{}\n"
          "------------------------------confidence------\t\t------equality------\n"
          "Train cla Accuracy       || {}\t\t{}\n"
          "Test cla Accuracy        || {}\t\t{}\n"
          "Valid cla Accuracy       || {}\t\t{}\n".format
          (result[0],
           result[1], result[3],
           result[5], result[7],
           result[9], result[11],
           result[2], result[4],
           result[6], result[8],
           result[10], result[12]))


def normalized_train(data):
    return data / len(hj_generate.train_dataloader)


def normalized_loss(data):  # nllloss has divide the batch, thus should use the length of dataloader
    return data / len(hj_generate.train_dataloader)


def normalized_test(data):
    return data / len(hj_generate.test_dataloader)


def normalized_valid(data):
    return data / len(hj_generate.valid_dataloader)


def fine_turning_model(model):
    model.dense = MultiTaskModel().dense
    for k, i in model.named_parameters():
        if ("conv4" not in k) and ('dense' not in k):
            i.requires_grad = False
    # for i in model.parameters():
    #     print(i.requires_grad)
    return model


def infer_data_process(model, data_loader):
    infer_correct_obj = 0
    infer_correct_cla = 0
    _infer_correct_obj = 0
    _infer_correct_cla = 0
    for data, label in data_loader:
        x_infer, y_infer = data, label
        x_infer, y_infer = Variable(x_infer).to(device), Variable(y_infer).to(device)
        outputs = model(x_infer)

        # obtain output
        out_obj = outputs[:, :2]
        out_cla = outputs[:, 2:]
        value_obj, pred_obj = torch.max(out_obj.data, 1)
        value_cla, pred_cla = torch.max(out_cla.data, 1)

        # calculate correct value
        _value_obj = 0
        _value_cla = 0
        value_obj = torch.sum(value_obj * (pred_obj == y_infer[:, 0].data)) / label.shape[0]
        _value_obj = torch.sum(pred_obj == y_infer[:, 0].data) / label.shape[0]
        if n := len([*filter(lambda x: x, label[:, 0])]):
            # confidence
            value_cla = torch.sum(value_cla * (pred_cla == y_infer[:, 1].data)) / n
            # equality
            _value_cla = torch.sum(pred_cla == y_infer[:, 1].data) / n
        else:
            value_cla = 1

        # accumulation
        infer_correct_obj += value_obj
        infer_correct_cla += value_cla
        _infer_correct_obj += _value_obj
        _infer_correct_cla += _value_cla
    return float(infer_correct_obj) / len(data_loader), float(infer_correct_cla) / len(data_loader), \
           float(_infer_correct_obj) / len(data_loader), float(_infer_correct_cla) / len(data_loader)


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
        content += "\t\t  train_oc"
        content += "\t\t  train_oe"
        content += "\t\t  train_cc"
        content += "\t\t  train_ce"
        content += "\t\t  test_oc"
        content += "\t\t  test_oe"
        content += "\t\t  test_cc"
        content += "\t\t  test_ce"
        content += "\t\t  valid_oc"
        content += "\t\t  valid_oe"
        content += "\t\t  valid_cc"
        content += "\t\t  valid_ce"
        content += "\n"
        for result in train_result:
            content += ("\t\t  " + str("{:5f}".format(result)))
        content += '\n'
        f_txt.write(content)
    f_txt.close()


def export_train_data_gpt(train_result):
    train_output = "../weight/" + file_name + '/'
    if not os.path.exists(train_output):
        os.mkdir(train_output)
    with open(train_output + 'train_result.txt', 'a+') as f:
        memories = int(len(f.readlines()) / 2)
        content = f"{memories + 1}\t\t  loss_val\t\t  train_oc\t\t  train_oe\t\t  train_cc\t\t  " \
                  f"train_ce\t\t  test_oc\t\t  test_oe\t\t  test_cc\t\t  test_ce\t\t  " \
                  f"valid_oc\t\t  valid_oe\t\t  valid_cc\t\t  valid_ce\n"
        content += "\t\t  "
        content += "\t\t  ".join([f"{result:5f}" for result in train_result])
        content += '\n'
        f.write(content)


def train():
    # load saved model
    model = torch.load(model_path, map_location=device)

    # fine_turning model from pre train model
    # model = fine_turning_model(torch.load(pre_train_path, map_location=device))

    # train a new model
    # model = MultiTaskModel()

    # model.load_state_dict(torch.load(model_path))
    epochs = 10
    lr = 0.001
    loss_fn = CrossEntropyLoss()  # nllloss(log_softmax)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005)
    # optimizer=SGD(model.parameters(),lr=lr,weight_decay=0.09,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print(model)
    model = model.to(device)

    best_model = MultiTaskModel()
    best_accuracy = 0
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch + 1, epochs))
        print('-' * 10)
        running_loss = 0.0
        running_correct_obj = 0
        running_correct_cla = 0
        _running_correct_obj = 0
        _running_correct_cla = 0
        model.train()
        for data, label in hj_generate.train_dataloader:
            X_train, y_train = data, label
            X_train, y_train = Variable(X_train).to(device), Variable(y_train).to(device)
            outputs = model(X_train)

            # get output
            out_obj = outputs[:, :2]
            out_cla = outputs[:, 2:]
            value_obj, pred_obj = torch.max(out_obj.data, 1)
            value_cla, pred_cla = torch.max(out_cla.data, 1)

            # loss backward
            optimizer.zero_grad()
            loss_obj = torch.zeros(1, device=device)
            loss_cla = torch.zeros(1, device=device)
            loss_total = torch.zeros(1, device=device)
            # for multi-task
            for out_obj_ele, out_cla_ele, y_train_ele in zip(out_obj, out_cla, y_train):
                loss_obj += loss_fn(out_obj_ele, y_train_ele[0].long())
                if y_train_ele[0]:
                    loss_cla += loss_fn(out_cla_ele, y_train_ele[1].long())
                loss_total += (1.0 * loss_obj + 1.4 * loss_cla)

            loss_total /= label.shape[0]
            loss_total.backward()
            optimizer.step()

            # calculate correct value
            _value_obj = 0
            _value_cla = 0
            value_obj = torch.sum(value_obj * (pred_obj == y_train[:, 0].data)) / label.shape[0]
            _value_obj = torch.sum(pred_obj == y_train[:, 0].data) / label.shape[0]
            if n := len([*filter(lambda x: x, label[:, 0])]):
                # confidence
                value_cla = torch.sum(value_cla * (pred_cla == y_train[:, 1].data)) / n
                # equality
                _value_cla = torch.sum(pred_cla == y_train[:, 1].data) / n
            else:
                value_cla = 1

            running_loss += float(loss_total.data)  # data cannot compute gradient
            running_correct_obj += value_obj
            running_correct_cla += value_cla
            _running_correct_obj += _value_obj
            _running_correct_cla += _value_cla

        # implement in train set
        # training_pre_correct_obj, training_pre_correct_cla, _training_pre_correct_obj, _training_pre_correct_cla \
        #     = infer_data_process(model, hj_generate.test_dataloader)

        # implement in test set
        testing_pre_correct_obj, testing_pre_correct_cla, _testing_pre_correct_obj, _testing_pre_correct_cla \
            = infer_data_process(model, hj_generate.test_dataloader)

        # implement in valid set
        valid_pre_correct_obj, valid_pre_correct_cla, _valid_pre_correct_obj, _valid_pre_correct_cla \
            = infer_data_process(model, hj_generate.valid_dataloader)

        # training loss
        training_pre_loss = normalized_loss(running_loss)

        # object distinguish accuracy(confidence)
        training_pre_correct_obj = normalized_train(running_correct_obj)

        # classical distinguish accuracy(confidence)
        training_pre_correct_cla = normalized_train(running_correct_cla)

        # object distinguish accuracy(equality)
        _training_pre_correct_obj = normalized_train(_running_correct_obj)

        # classical distinguish accuracy(equality)
        _training_pre_correct_cla = normalized_train(_running_correct_cla)

        if 0.7 * (0.2 * training_pre_correct_obj + 0.3 * testing_pre_correct_obj + 0.5 * valid_pre_correct_obj) + \
                0.3 * (0.2 * training_pre_correct_cla + 0.3 * testing_pre_correct_cla + 0.5 * valid_pre_correct_cla) \
                > best_accuracy:
            print("more excellent!!!")
            best_accuracy = \
                0.7 * (0.2 * training_pre_correct_obj + 0.3 * testing_pre_correct_obj + 0.5 * valid_pre_correct_obj) + \
                0.3 * (0.2 * training_pre_correct_cla + 0.3 * testing_pre_correct_cla + 0.5 * valid_pre_correct_cla)
            best_model = model
        result = [training_pre_loss,
                  training_pre_correct_obj, training_pre_correct_cla, _training_pre_correct_obj,
                  _training_pre_correct_cla,
                  testing_pre_correct_obj, testing_pre_correct_cla, _testing_pre_correct_obj, _testing_pre_correct_cla,
                  valid_pre_correct_obj, valid_pre_correct_cla, _valid_pre_correct_obj, _valid_pre_correct_cla]
        show(result)
        export_train_data(result)

        scheduler.step()
    torch.save(best_model, save_path)
    # torch.onnx.export(best_model, input_data, save_path, opset_version=9, verbose=True, input_names=input_names,
    #                   output_names=output_names, dynamic_axes={'input': {0: '1'}}, )#export model direct


if __name__ == "__main__":
    # train()
    export_train_data_gpt([1, 2, 3, 4, 5, 4, 3, 4, 3, 3, 3, 3, 3])
