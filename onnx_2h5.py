import onnx
from onnx2keras import onnx_to_keras
import tensorflow.keras as keras

file_name='2023_2_1_hj_num_1'
input_path = './onnx_model/'+file_name+'.onnx'
output_path = './h5_model/'+file_name+'.h5'

def onnx2h5():
    onnx_model = onnx.load(input_path)
    k_model = onnx_to_keras(onnx_model, ['input'],name_policy="renumerate")
    keras.models.save_model(k_model, output_path, overwrite=True, include_optimizer=True)  # 第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = keras.models.load_model(output_path)
    model.summary()
    print(model)

if __name__ == '__main__':
    onnx2h5()