import hiddenlayer as hl
import time
# from hj_num_v4_5 import model_path
import numpy
import torch
import numpy as np

file_name = '2023_4_4_hj_num_1'
new_name = '2023_4_4_hj_num_1'
pre_train_name = '2023_3_11_hj_num_1'
model_path = '../weight/' + file_name + '.pt'


def hl_visualize_multitask(epoch, step, train_loss, valid_obj_acc, valid_cla_acc, model, history, canvas):
    history.log((epoch, step),
                train_loss=train_loss,
                valid_obj_acc=valid_obj_acc,
                valid_cla_acc=valid_cla_acc)

    with canvas:
        canvas.draw_plot(history["train_loss"])
        canvas.draw_plot(history["valid_obj_acc"])
        canvas.draw_plot(history["valid_cla_acc"])
        # canvas.draw_plot(history["conv4_2_weight"])


def hl_visualize(epoch, step, train_loss, train_acc, model, history, canvas):
    history.log((epoch, step),
                train_loss=train_loss,
                train_acc=train_acc,)

    with canvas:
        canvas.draw_plot(history["train_loss"])
        canvas.draw_plot(history["train_acc"])
        # canvas.draw_plot(history["valid_acc"])
        # canvas.draw_plot(history["conv4_2_weight"])


if __name__ == "__main__":
    history = hl.History()
    canvas = hl.Canvas()
    model = torch.load(model_path)
    epoch = 10
    step = 100
    batches = 0
    for i in range(1000):
        batches += 1
        if batches % step == 0:
            history.log((epoch, batches),
                        conv4_2_weight=(model.dense[4].weight.to("cpu")[0]))

            with canvas:
                canvas.draw_plot(history["conv4_2_weight"])
