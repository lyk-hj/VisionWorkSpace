import torch.onnx
from hj_num_v4 import Model
from onnxmltools.utils.float16_converter import convert_float_to_float16
# from onnxconverter_common import convert_tensor_float_to_float16
from onnxconverter_common.float16 import convert_np_to_float16
import numpy as np
import onnx,os

fp16 = True
dynamic = True
file_name='2023_3_16_hj_num_1'
model_path = './weight/'+file_name+'.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

def convert_fp16(save_path):
    onnx_model = onnx.load_model(save_path)
    fp16_onnx_model = convert_float_to_float16(onnx_model, keep_io_types=False)
    onnx.save_model(fp16_onnx_model,save_path)

def export(model, save_path):
    input_data = torch.randn(1, 1, 24, 24, device=device)

    # 转化为onnx模型
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, input_data, save_path, opset_version=9, verbose=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes= {'input':{0:'1'}} if 'dyn' in save_path else {})
    if 'fp16' in save_path:
        convert_fp16(save_path)

def pt_2onnx():
    print(device)
    model = torch.load(model_path)

    model.to(device)
    model.eval()

    initial_path = './onnx_model/' + file_name + '/'
    if not os.path.exists(initial_path):
        os.mkdir(initial_path)

    initial_path += file_name
    export(model, initial_path + '.onnx')
    if dynamic:
        save_path = initial_path + '_dyn'
        export(model, save_path + '.onnx')
    if fp16:
        save_path = initial_path + '_fp16'
        export(model, save_path + '.onnx')
    if dynamic and fp16:
        save_path = initial_path + '_dyn_fp16'
        export(model, save_path + '.onnx')




if __name__ == "__main__":
    pt_2onnx()