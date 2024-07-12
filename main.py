import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import subprocess
from tqdm import tqdm
from config_physnet import BACKBONES, BACKBONE_KWARGS  # 确保这个模块在同一目录下

# 模型和数据集的路径设置
ROOT = "/Users/jason/Desktop/hpc_model/ai4molcryst_argonne/physnet_model"
SAVE_ROOT = os.path.join(ROOT, "pretrained_models")
os.makedirs(SAVE_ROOT, exist_ok=True)

# 模型名和任务
opt_model_name = "physnet"
opt_dataset = "qm9"
opt_task = "homo"  # 如 homo, lumo 等

# 下载模型权重
opt_name = f"{opt_model_name}_pub_{opt_dataset}{opt_task}"
model_path = os.path.join(SAVE_ROOT, f"{opt_name}.pth")
if not os.path.exists(model_path):
    cmd = f"wget https://zenodo.org/record/7758490/files/{opt_name}.pth?download=1 -O {model_path}"
    proc = subprocess.run(cmd, capture_output=True, shell=True)
    if proc.returncode != 0:
        print("Error downloading the model:", proc.stderr.decode())
    else:
        print("Model downloaded successfully.")
else:
    print("Model already exists.")

# 加载模型
model_kwargs = BACKBONE_KWARGS[opt_model_name]
model = BACKBONES[opt_model_name](**model_kwargs)
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt['model'])
    model.eval()  # 将模型设置为评估模式
    print(f"{model.__class__.__name__} is loaded with {opt_dataset.upper()} {opt_task.upper()}")

# 加载数据集
data_root = os.path.join(ROOT, 'qm9/dataset')  # 设置数据集的存储路径
dataset = QM9(root=data_root)

# 测试数据集索引列表
test_indices = [index for index in range(len(dataset)) if index % 10 == 0]
test_dataset = dataset[test_indices]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 测试模型性能
predictions, targets = [], []
for data in tqdm(test_loader, desc="Testing Model"):
    out = model(z=data.z, pos=data.pos, batch=data.batch)
    predictions.append(out[0].detach())  # 假设out是一个元组，且我们需要第一个元素
    if data.y.dim() > 1 and data.y.shape[1] > 1:
        targets.append(data.y[:, 0].detach())  # 只取第一个属性
    else:
        targets.append(data.y.detach())

flat_predictions = torch.cat(predictions, dim=0)
flat_targets = torch.cat(targets, dim=0)
mae = mean_absolute_error(flat_targets.numpy(), flat_predictions.numpy())
rmse = mean_squared_error(flat_targets.numpy(), flat_predictions.numpy())
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
