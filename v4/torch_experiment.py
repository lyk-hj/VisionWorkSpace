import torch
import numpy as np
import random
from model_v4 import Model

file_name = '2023_4_9_hj_num_1'
model_path = '../weight/' + file_name + '.pt'

def model_experiment():
    a = torch.load(model_path)
    b = a
    a = Model()
    print(*b.conv0.parameters())
    print(*a.conv0.parameters())


if __name__ == "__main__":
    model_experiment()