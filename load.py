# # # 加载Torch Script模型（.pt）
import torch

# 加载Torch Script模型
scripted_model = torch.jit.load("model/best_model_scripted.pt")
scripted_model.eval()  # 设置为评估模式

# 示例推理
example_input = torch.rand(1, 398)  # 示例输入
with torch.no_grad():
    output = scripted_model(example_input)
    print("Torch Script模型输出:", output)


# # # 加载ONNX模型（.onnx）
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
ort_session = ort.InferenceSession("model/best_model.onnx")

# 准备输入数据
input_name = ort_session.get_inputs()[0].name
example_input = np.random.rand(1, 398).astype(np.float32)  # 示例输入

# 推理
outputs = ort_session.run(None, {input_name: example_input})
print("ONNX模型输出:", outputs)
