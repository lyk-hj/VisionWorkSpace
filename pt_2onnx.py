import torch.onnx
from hj_num_v4 import Model


# 创建.pth模型

# model = Model()
# 加载权重
file_name='2023_3_15_hj_num_2'
model_path = './weight/'+file_name+'.pt'
save_path = './onnx_model/'+file_name+'.onnx'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torch.load(model_path)

model.to(device)
model.eval()

input_data = torch.randn(1, 1, 24, 24, device=device)

# 转化为onnx模型
input_names = ['input']
output_names = ['output']

torch.onnx.export(model, input_data, save_path, opset_version=9, verbose=True, input_names=input_names,
                  output_names=output_names, dynamic_axes={'input':{0:'1'}},)
