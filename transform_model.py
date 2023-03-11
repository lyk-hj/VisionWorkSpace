import onnx
from onnx2keras import onnx_to_keras
import tensorflow.keras as keras
import torch
from typing import Callable,Any

file_name='2023_1_31_hj_num_1'
input_path = './onnx_model/'+file_name+'.onnx'

class TransformModel:
    def __init__(self,need_model_type,input_path):
        self.input_path = input_path

        flag = 0
        start = 0
        end = 0
        for num,i in enumerate(input_path):
            if i == '/':
                flag=1
                continue
            elif i == '.':
                flag=2
            if flag == 1:
                start = num
                flag=0
            elif flag == 2:
                end = num
                flag=0
        self.model_name = input_path[start:end]
        # print(self.model_name)

        if need_model_type == "h5":
            if input_path[end:] == ".onnx":
                output_path = "./h5_model/" + self.model_name + ".h5"
                self.onnx2h5(output_path)
        elif need_model_type == "onnx":
            if input_path[end:] == ".pt" or input_path[end:] == ".pth":
                output_path = "./onnx_model/" + self.model_name + ".onnx"
                self.pt2onnx(output_path)
    def onnx2h5(self,output_path):
        onnx_model = onnx.load(self.input_path)
        k_model = onnx_to_keras(onnx_model, ['input'])
        keras.models.save_model(k_model, output_path, overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
        # 下面内容是加载该模型，然后将该模型的结构打印出来
        model = keras.models.load_model(output_path,compile=False)
        model.summary()
        print(output_path)
        print(model)

    def pt2onnx(self,output_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_statedict = torch.load(self.input_path, map_location=device)

        model_statedict.to(device)
        model_statedict.eval()

        input_data = torch.randn(1, 1, 30, 20, device=device)

        # 转化为onnx模型
        input_names = ['input']
        output_names = ['output']

        print(output_path)
        torch.onnx.export(model_statedict, input_data, output_path, opset_version=9, verbose=True,
                          input_names=input_names,
                          output_names=output_names)

    def printnum(self,data):
        print(data)
        return data*3

    __call__ : Callable[...,Any] = printnum


if __name__ == "__main__":
    transformTool = TransformModel("h5",input_path)
    data = transformTool(8)
    print(data)