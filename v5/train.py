import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW
from torch.autograd import Variable
from model_v5 import ClsModel
import generate as hj_generate
import os
import random
import hiddenlayer as hl
# from visualize_train import hl_visualize
import copy

file_name = '2023_4_14_hj_num_1'
new_name = '2023_4_14_hj_num_1'
pre_train_name = '2023_3_11_hj_num_1'
model_path = '../weight/' + file_name + '.pt'
save_path = '../weight/' + new_name + '.pt'
pre_train_path = '../weight/' + pre_train_name + '.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return data / len(hj_generate.train_dataloaders[0])


def normalized_test(data):
    return data / len(hj_generate.test_dataloader)


def normalized_valid(data):
    return data / len(hj_generate.valid_dataloader)


def fine_turning_model(model):
    model.dense = ClsModel().dense
    for k, i in model.named_parameters():
        if ("conv4" not in k) and ('dense' not in k):
            i.requires_grad = False
    # for i in model.parameters():
    #     print(i.requires_grad)
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
    epochs = 10
    lr = 0.001
    log_step = 100
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.005)
    # optimizer=SGD(model.parameters(),lr=lr,weight_decay=0.09,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    print(model)
    model = model.to(device)

    best_model = copy.deepcopy(model)
    best_accuracy = 0
    history = hl.History()
    canvas = hl.Canvas()
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch + 1, epochs))
        print('-' * 10)
        running_loss = 0.0
        running_correct = 0
        _running_correct = 0
        model.train()

        for batch_step, (data, label) in enumerate(random.choice(hj_generate.train_dataloaders)):
            batch_step += 1
            X_train, y_train = data, label
            X_train, y_train = Variable(X_train).to(device), Variable(y_train).to(device)
            outputs = model(X_train)
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
                            train_acc=training_pre_correct)

                with canvas:
                    canvas.draw_plot(history["train_loss"])
                    canvas.draw_plot(history["train_acc"])
                print('-' * 10)
                # hl_visualize(epoch, log_step,
                #              training_pre_loss,
                #              training_pre_correct,
                #              model, history, canvas)

        # implement in test set
        testing_pre_correct,  _testing_pre_correct \
            = infer_data_process(model, hj_generate.test_dataloader)

        # implement in valid set
        valid_pre_correct, _valid_pre_correct \
            = infer_data_process(model, hj_generate.valid_dataloader)

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


if __name__ == "__main__":
    cls_train()