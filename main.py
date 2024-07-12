import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import subprocess
from tqdm import tqdm
from config_physnet import BACKBONES, BACKBONE_KWARGS

# Set the ROOT path
ROOT = os.getcwd()
SAVE_ROOT = os.path.join(ROOT, "pretrained_models")
os.makedirs(SAVE_ROOT, exist_ok=True)

# model and task
opt_model_name = "physnet"
opt_dataset = "qm9"
opt_task = "homo"  # 如 homo, lumo 等

# download the weights
opt_name = f"{opt_model_name}_pub_{opt_dataset}{opt_task}"
model_path = os.path.join(SAVE_ROOT, f"{opt_name}.pth")
if not os.path.exists(model_path):
    cmd = f"wget https://zenodo.org/record/7758490/files/{
        opt_name}.pth?download=1 -O {model_path}"
    proc = subprocess.run(cmd, capture_output=True, shell=True)
    if proc.returncode != 0:
        print("Error downloading the model:", proc.stderr.decode())
    else:
        print("Model downloaded successfully.")
else:
    print("Model already exists.")

# load the model
model_kwargs = BACKBONE_KWARGS[opt_model_name]
model = BACKBONES[opt_model_name](**model_kwargs)
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(ckpt['model'])
    model.eval()  # eval mode
    print(f"{model.__class__.__name__} is loaded with {
          opt_dataset.upper()} {opt_task.upper()}")

# load dataset
data_root = os.path.join(ROOT, 'qm9/dataset')  # save path
dataset = QM9(root=data_root)

# get test dataset
test_indices = [index for index in range(len(dataset)) if index % 10 == 0]
test_dataset = dataset[test_indices]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# test the model
predictions, targets = [], []
for data in tqdm(test_loader, desc="Testing Model"):
    out = model(z=data.z, pos=data.pos, batch=data.batch)
    predictions.append(out[0].detach())
    if data.y.dim() > 1 and data.y.shape[1] > 1:
        targets.append(data.y[:, 0].detach())
    else:
        targets.append(data.y.detach())

flat_predictions = torch.cat(predictions, dim=0)
flat_targets = torch.cat(targets, dim=0)
mae = mean_absolute_error(flat_targets.numpy(), flat_predictions.numpy())
rmse = mean_squared_error(flat_targets.numpy(), flat_predictions.numpy())
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
