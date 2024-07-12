# 检查 PyTorch 和相关库的安装情况
import torchvision
import torchaudio
import torch_cluster
import torch_geometric
import torch_scatter
import torch_sparse

# 打印各个库的版本号
print("Torchvision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
print("Torch Cluster version:", torch_cluster.__version__)
print("Torch Geometric version:", torch_geometric.__version__)
print("Torch Scatter version:", torch_scatter.__version__)
print("Torch Sparse version:", torch_sparse.__version__)


print("OK")
