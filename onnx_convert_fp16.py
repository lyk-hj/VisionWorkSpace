# this module just use to convert onnx model to fp16 precision and infer on OpenCV DNN or PyTorch,
# normally recommend to use fp32 precision model
# as other neural network inference tools own their individual precision transformer

from onnxmltools.utils.float16_converter import convert_float_to_float16
# from onnxconverter_common import convert_tensor_float_to_float16
from onnxconverter_common.float16 import convert_np_to_float16
import numpy as np
import onnx

file_name='2023_2_26_hj_num_2'
model_path = './onnx_model/'+file_name+'.onnx'
save_path = './onnx_model/'+file_name+'_fp16.onnx'

onnx_model = onnx.load_model(model_path)
# float32_list = np.fromstring(onnx_model.raw_data, dtype='float32')
# float16_list = convert_np_to_float16(float32_list)
# onnx_model.raw_data = float16_list.tostring()
fp16_onnx_model = convert_float_to_float16(onnx_model, keep_io_types=False)
onnx.save_model(fp16_onnx_model,save_path)