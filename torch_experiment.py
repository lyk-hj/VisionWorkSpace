import torch
import numpy as np
import random

def multi_task_experiment():
    output = torch.Tensor([[0, 1, 0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0, 0, 0], [5, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                           [5, 0, 0, 0, 0, 1, 1, 0, 0, 0], [5, 0, 0, 0, 0, 1, 1, 0, 0, 0]])
    out = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 10, 0, 0, 0, 0]])
    loss_fn = torch.nn.CrossEntropyLoss()
    label = torch.tensor([[1,2], [1,3], [0, -1], [0, -1], [0, -1]])
    lab = torch.tensor([2,3])
    # l = torch.tensor([-1,1])
    # one_hot = F.one_hot(label).float()
    # one_h = F.one_hot(lab).float()
    # one_ = F.one_hot(l).float()
    # print(one_hot)
    # print(one_)
    # log_softmax = torch.log(torch.softmax(output,1))
    # nll_loss = -torch.sum(one_hot*log_softmax)/label.shape[0] # label.shape[0] is batch
    # print(nll_loss/2)
    out_obj = output[:, :2]
    out_cla = output[:, 2:]
    value_obj, pred_obj = torch.max(out_obj.data, 1)
    value_cla, pred_cla = torch.max(out_cla.data, 1)

    # print(out_obj)
    # print(out_cla)
    # print(label[:, 1])
    # loss_cla = loss_fn(out_cla, label[:, 1].long())
    # loss_obj = loss_fn(out_obj, label[:, 0].long())
    # _loss = loss_fn(out, lab.long())
    # print(loss_cla.data)
    # print(_loss.data)
    # print(loss_obj)

    loss_obj = torch.zeros(1, device="cpu")
    loss_cla = torch.zeros(1, device="cpu")
    for out_obj_ele, out_cla_ele, y_train_ele in zip(out_obj, out_cla, label):
        loss_obj += loss_fn(out_obj_ele, y_train_ele[0].long())
        if y_train_ele[0]:
            loss_cla += loss_fn(out_cla_ele, y_train_ele[1].long())
    print(label.shape[0])
    print(loss_obj/label.shape[0])
    print(loss_cla/label.shape[0])
    print(*zip(out_obj,out_cla))
    print(value_obj, pred_obj)
    print(value_cla, pred_cla)
    print(len([*filter(lambda x: x, label[:, 0])]))
    a = torch.tensor(222.2)
    print(np.float64(a.data))
    if n:=1:
        print(n/2)
    else:
        print(3)
    # value, pred = torch.max(output.data, 1)
    # print(pred)
    # print(value)
    # for data, label in test_dataloader:
    #     print(data)
    #     print(label.data)
def random_experiment():
    a = [1,2,3,4,5]
    print(random.choices(a,k=10))

def copy_experiment():
    a = [1,2,5]
    b = a
    a.append(5)
    print(a, b)


if __name__ == "__main__":
    copy_experiment()