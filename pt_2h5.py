from pytorch2keras.converter import pytorch_to_keras
from torch.autograd import Variable
import torch
import tensorflow.keras as keras
import numpy as np

file_name='2023_2_1_hj_num_2'
input_path = './weight/'+file_name+'.pt'
output_path = './h5_model/'+file_name+'.h5'

def pt2h5():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    pt_model = torch.load(input_path)
    pt_model.to(device)
    pt_model.eval()
    # Make dummy variables (and checking if the model works)
    input_np = np.random.uniform(0, 1, (1, 1, 30, 20))
    input_var = Variable(torch.FloatTensor(input_np)).to(device)
    output = pt_model(input_var)

    # Convert the model!
    k_model = \
        pytorch_to_keras(pt_model, input_var, (1, 30, 20),
                         verbose=True, name_policy='renumerate')

    # Save model to SavedModel format
    # tf.saved_model.save(k_model, "./models")
    keras.models.save_model(k_model, output_path, overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = keras.models.load_model(output_path)
    model.summary()
    print(model)

if __name__=="__main__":
    pt2h5()