import torch.onnx
from onnxmltools.utils.float16_converter import convert_float_to_float16
import onnx, os
from model_v4 import MultiTaskModel
from model_v4 import Model
from pytorch2keras.converter import pytorch_to_keras
from torch.autograd import Variable
import torch
import tensorflow.keras as keras
import numpy as np

fp16 = True
dynamic = True
file_name='2023_4_21_hj_num_1'
model_path = '../weight/'+file_name+'.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def convert_fp16(save_path):
    onnx_model = onnx.load_model(save_path)
    fp16_onnx_model = convert_float_to_float16(onnx_model, keep_io_types=False)
    onnx.save_model(fp16_onnx_model,save_path)


def export(model, save_path):
    # python rules height before width, while opencv rules width before height
    input_data = torch.randn(1, 1, 30, 22, device=device)

    # 转化为onnx模型
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, input_data, save_path, opset_version=9, verbose=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes={'input': {0: '1', 2: 'height', 3: 'width'}} if 'dyn' in save_path else None)
    if 'fp16' in save_path:
        convert_fp16(save_path)


def pt_2onnx():
    model = torch.load(model_path)

    model.to(device)
    model.eval()

    initial_path = '../onnx_model/' + file_name + '/'
    if not os.path.exists(initial_path):
        os.mkdir(initial_path)

    initial_path += file_name
    export(model, initial_path + '.onnx')
    if dynamic:
        trans_path = initial_path + '_dyn'
        export(model, trans_path + '.onnx')
    if fp16:
        trans_path = initial_path + '_fp16'
        export(model, trans_path + '.onnx')
    if dynamic and fp16:
        trans_path = initial_path + '_dyn_fp16'
        export(model, trans_path + '.onnx')


def pt2h5():
    output_path = '../h5_model/' + file_name + '.h5'
    pt_model = torch.load(model_path)
    pt_model.to(device)
    pt_model.eval()
    # Make dummy variables (and checking if the model works)
    input_np = np.random.uniform(0, 1, (1, 1, 30, 22))
    input_var = Variable(torch.FloatTensor(input_np)).to(device)
    output = pt_model(input_var)

    # Convert the model!
    k_model = \
        pytorch_to_keras(pt_model, input_var, (1, 30, 22),
                         verbose=True, name_policy='renumerate')

    # Save model to SavedModel format
    # tf.saved_model.save(k_model, "./models")
    keras.models.save_model(k_model, output_path, overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = keras.models.load_model(output_path)
    model.summary()
    print(model)


if __name__=="__main__":
    pt_2onnx()
