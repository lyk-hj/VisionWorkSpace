import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import AdamW
from torch.autograd import Variable
from model_v4 import Model
import hj_generate_v4 as hj_generate
import os


file_name = '2023_3_27_hj_num_2'
new_name = '2023_3_27_hj_num_2'
pre_train_name = '2023_3_11_hj_num_1'
model_path = '../weight/' + file_name + '.pt'
save_path = '../weight/' + new_name + '.pt'
pre_train_path = '../weight/' + pre_train_name + '.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def show(train_loss,
         train_acc,
         test_acc,
         valid_acc,
         traditional=False):
    if (traditional):
        print("Loss                     || {}\n"
              "Train old Accuracy       || {}\n"
              "Test old Accuracy        || {}\n"
              "Valid old Accuracy       || {}\n".format
              (train_loss,
               train_acc,
               test_acc,
               valid_acc))
    else:
        print("Loss                     || {}\n"
              "Train new Accuracy       || {}\n"
              "Test new Accuracy        || {}\n"
              "Valid new Accuracy       || {}\n".format
              (train_loss,
               train_acc,
               test_acc,
               valid_acc))


def normalized_train(data):
    return data / len(hj_generate.train_dataset)


def normalized_test(data):
    return data / len(hj_generate.test_dataset)


def normalized_valid(data):
    return data / len(hj_generate.valid_dataset)


def fine_turning_model(model):
    model.dense = Model().dense
    for k,i in model.named_parameters():
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
            memories = int(len(n_txt.readlines())/2)
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

def train():
    # load saved model
    # model = torch.load(model_path, map_location=device)

    # fine_turning model from pre train model
    # model = fine_turning_model(torch.load(pre_train_path, map_location=device))

    # train a new model
    model = Model()

    # model.load_state_dict(torch.load(model_path))
    epochs = 15
    lr = 0.001
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()), lr=lr, weight_decay=0.0005)
    # optimizer=SGD(model.parameters(),lr=lr,weight_decay=0.09,momentum=0.99)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    print(model)
    model = model.to(device)

    best_model = Model()
    best_accuracy = 0
    for epoch in range(epochs):
        print("Epoch:{}/{}".format(epoch + 1, epochs))
        print('-' * 10)
        running_loss = 0.0
        running_correct_new = 0
        running_correct_old = 0
        model.train()
        for data, label in hj_generate.train_dataloader:
            X_train, y_train = data, label
            X_train, y_train = Variable(X_train).to(device), Variable(y_train).to(device)
            outputs = model(X_train)
            value, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = loss_fn(outputs, y_train.long())
            loss.backward()
            optimizer.step()
            value = value * (pred == y_train.data)
            running_loss += loss.data
            running_correct_old += torch.sum(pred == y_train.data)
            running_correct_new += torch.sum(value)
        testing_correct_old = 0
        testing_correct_new = 0
        model.eval()
        for data, label in hj_generate.test_dataloader:
            X_test, y_test = data, label
            X_test, y_test = Variable(X_test).to(device), Variable(y_test).to(device)
            outputs = model(X_test)
            value, pred = torch.max(outputs.data, 1)
            value = value * (pred == y_test.data)
            testing_correct_old += torch.sum(pred == y_test.data)
            testing_correct_new += torch.sum(value)

        valid_correct_old = 0
        valid_correct_new = 0
        for data, label in hj_generate.valid_dataloader:
            X_check, y_check = data, label
            X_check, y_check = Variable(X_check).to(device), Variable(y_check).to(device)
            outputs = model(X_check)
            value, pred = torch.max(outputs.data, 1)
            value = value * (pred == y_check.data)
            valid_correct_old += torch.sum(pred == y_check.data)
            valid_correct_new += torch.sum(value)

        # obsolete traditional training data
        running_pre_loss = normalized_train(running_loss)
        running_pre_correct_old = normalized_train(running_correct_old)
        testing_pre_correct_old = normalized_test(testing_correct_old)
        valid_pre_correct_old = normalized_valid(valid_correct_old)

        # individual training data
        running_pre_correct_new = normalized_train(running_correct_new)
        testing_pre_correct_new = normalized_test(testing_correct_new)
        valid_pre_correct_new = normalized_valid(valid_correct_new)

        if 0.2 * running_pre_correct_new + 0.3 * testing_pre_correct_new + 0.5 * valid_pre_correct_new > best_accuracy:
            print("more excellent!!!")
            best_accuracy = 0.2 * running_pre_correct_new + 0.3 * testing_pre_correct_new + 0.5 * valid_pre_correct_new
            best_model = model

        show(running_pre_loss,
             running_pre_correct_old,
             testing_pre_correct_old,
             valid_pre_correct_old,
             True)

        show(running_pre_loss,
             running_pre_correct_new,
             testing_pre_correct_new,
             valid_pre_correct_new)

        scheduler.step()
    torch.save(best_model, save_path)
    # torch.onnx.export(best_model, input_data, save_path, opset_version=9, verbose=True, input_names=input_names,
    #                   output_names=output_names, dynamic_axes={'input': {0: '1'}}, )#export model direct


if __name__ == "__main__":
    train()

