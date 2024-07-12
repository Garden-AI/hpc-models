import torch
from sklearn import preprocessing
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools, functools, operator, collections
from typing import *
import networkx as nx
import einops
from einops.layers.torch import Rearrange, Reduce
from typing import *
import warnings
import os, glob, re, sys
import curtsies.fmtfuncs as cf
import copy
import tqdm
import pickle
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch_geometric.data import Dataset, download_url, extract_zip
from torch_geometric.data.makedirs import makedirs
from torch_geometric.nn import MessagePassing, radius_graph
import os.path as osp
import warnings
from math import pi as PI
from typing import Optional
import warnings
warnings.simplefilter("ignore")

qm9_target_dict = {
    0: 'dipole_moment',
    1: 'isotropic_polarizability',
    2: 'homo',
    3: 'lumo',
    4: 'gap',
    5: 'electronic_spatial_extent',
    6: 'zpve',
    7: 'energy_U0',
    8: 'energy_U',
    9: 'enthalpy_H',
    10: 'free_energy',
    11: 'heat_capacity',
}

import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory
from schnet import schnet
SchNet = schnet.SchNet

class SchNet(SchNet):
    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None, explain: bool=False):
        super().__init__(hidden_channels=hidden_channels, num_filters=num_filters,
                 num_interactions=num_interactions, num_gaussians=num_gaussians,
                 cutoff=cutoff, max_num_neighbors=max_num_neighbors,
                 readout=readout, dipole=dipole,
                 mean=mean, std=std,
                 atomref=atomref, explain=explain)
        self.embedding = torch.nn.Linear(92,hidden_channels)
        self.expand_nbr_fea = torch.nn.Linear(41, num_gaussians)
        self.readout = "mean"

        if self.explain:
            def hook(module, inputs, grad):
                self.embedded_grad = grad
            self.embedding.register_backward_hook(hook)
			
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch):

        h = self.embedding(atom_fea)
        row, col = nbr_fea_idx
        edge_weight = dists
        edge_attr = self.expand_nbr_fea(nbr_fea)

        for interaction in self.interactions:
            h = h + interaction(h, nbr_fea_idx, edge_weight, edge_attr)

        h = self.lin1(h) #num_nodes, dim
        
        if self.explain:
            self.final_conv_acts = h
            def hook(grad):
                self.final_conv_grads = grad
            self.final_conv_acts.register_hook(hook) #only when backpropped!	
            
        h = self.act(h)
        h = self.lin2(h)

#         if self.dipole:
#             # Get center of mass.
#             mass = self.atomic_mass[z].view(-1, 1)
#             c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
#             h = h * (pos - c.index_select(0, batch))

#         if not self.dipole and self.mean is not None and self.std is not None:
#             h = h * self.std + self.mean

#         if not self.dipole and self.atomref is not None:
#             h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

#         if self.dipole:
#             out = torch.norm(out, dim=-1, keepdim=True)

#         if self.scale is not None:
#             out = self.scale * out
            
        return out



    
    
if __name__ == "__main__":
	def get_parser():
		parser = argparse.ArgumentParser()
		#     parser.add_argument('--name', type=str, default=''.join(random.choice(string.ascii_lowercase) for i in range(10)))
		parser.add_argument('--name', type=str, default=None)
		parser.add_argument('--seed', type=int, default=7)
		parser.add_argument('--gpu', action='store_true')
		parser.add_argument('--gpus', action='store_true')
		parser.add_argument('--silent', action='store_true')
		parser.add_argument('--log', action='store_true') #only returns true when passed in bash
		parser.add_argument('--plot', action='store_true')
		parser.add_argument('--use-artifacts', action='store_true', help="download model artifacts for loading a model...") 

		# data
		parser.add_argument('--train_test_ratio', type=float, default=0.02)
		parser.add_argument('--train_val_ratio', type=float, default=0.03)
		parser.add_argument('--train_frac', type=float, default=0.8)
		parser.add_argument('--warm_up_split', type=int, default=5)
		parser.add_argument('--batches', type=int, default=160)
		parser.add_argument('--test_samples', type=int, default=5) # -1 for all
		parser.add_argument('--test_steps', type=int, default=100)
		parser.add_argument('--data_norm', action='store_true') #normalize energy???
		parser.add_argument('--dataset', type=str, default="qm9edge", choices=["qm9", "md17", "ani1", "ani1x", "qm9edge", "moleculenet"])
		parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/data")
		parser.add_argument('--task', type=str, default="homo")
		parser.add_argument('--pin_memory', type=bool, default=True) #causes CUDAMemory error;; asynchronously reported at some other API call
		parser.add_argument('--use_artifacts', action="store_true", help="use artifacts for resuming to train")
		parser.add_argument('--use_tensors', action="store_true") #for data, use DGL or PyG formats?

		# train
		parser.add_argument('--epoches', type=int, default=2)
		parser.add_argument('--batch_size', type=int, default=128) #Per GPU batch size
		parser.add_argument('--num_workers', type=int, default=4)
		parser.add_argument('--learning_rate','-lr', type=float, default=1e-4)
		parser.add_argument('--weight_decay', type=float, default=2e-5)
		parser.add_argument('--dropout', type=float, default=0)
		parser.add_argument('--resume', action='store_true')
		parser.add_argument('--distributed',  action="store_true")
		parser.add_argument('--low_memory',  action="store_true")
		parser.add_argument('--amp', action="store_true", help="floating 16 when turned on.")
		parser.add_argument('--loss_schedule', '-ls', type=str, choices=["manual", "lrannealing", "softadapt", "relobralo", "gradnorm"], help="how to adjust loss weights.")
		parser.add_argument('--with_force', type=bool, default=False)
		parser.add_argument('--optimizer', type=str, default='adam', choices=["adam","lamb","sgd","torch_adam"])
		parser.add_argument('--gradient_clip', type=float, default=None) 
		parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
		parser.add_argument('--shard', action="store_true", help="fairscale ShardedDDP") #fairscale ShardedDDP?
		parser.add_argument(
		"--not_use_env",
		default=False,
		action="store_false",
		help="Use environment variable to pass "
		"'local rank'. For legacy reasons, the default value is False. "
		"If set to True, the script will not pass "
		"--local_rank as argument, and will instead set LOCAL_RANK.",
		)

		# model
		parser.add_argument('--backbone', type=str, default='physnet', choices=["schnet","physnet","torchmdnet","alignn","dimenet","dimenetpp"])
		parser.add_argument('--load_ckpt_path', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/save/")
		parser.add_argument('--explain', type=bool, default=False, help="gradient hook for CAM...") #Only for Schnet.Physnet.Alignn WIP!

		# hyperparameter optim
		parser.add_argument('--resume_hp', action="store_true", help="resume hp search if discontinued...")
		parser.add_argument('--hp_savefile', type=str, default="results.csv", help="resume hp search from this file...")
		opt = parser.parse_args()
		return opt

	import string; import random; import argparse
	opt = get_parser()

	config = dict(hidden_channels=128, 
					 num_filters=128,
					 num_interactions=6,
					 num_gaussians=50,
					 cutoff=10.0,
					 max_num_neighbors=32,
					 readout='add',
					 dipole=False,
					 mean=None,
					 std=None,
					 atomref=None)
	m = SchNet(**config)

	from crystals.general_train_eval import load_state
	path_and_name = os.path.join("/Scr/hyunpark/ArgonneGNN/argonne_gnn/save", "{}.pth".format("cschnet"))
	load_state(m, None, None, path_and_name, model_only=True)
	m.eval()

	#Get Dataloader
	import crystals.cgcnn_data_utils as cdu
	root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax"
	dataset = cdu.CIFData(root_dir)
	dataloader = cdu.get_dataloader(dataset, shuffle=False, **{'pin_memory': opt.pin_memory, 'persistent_workers': False,
                                 'batch_size': opt.batch_size})
	li = iter(dataloader).next()
	
	atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, y = li.x, li.edge_attr, li.edge_index, li.cif_id, li.batch, li.edge_weight, li.y
	result = m(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch)
	print(result.view(-1), y)
	print(result.view(-1) - y)

# 	import train.dataloader
# 	dl = train.dataloader.DataModuleEdge(opt=opt)
# 	opt.dataset="qm9edge"
# 	opt.task="lumo"
# 	td = dl.train_dataloader()
# 	li=next(iter(td)) #dict
# 	E = li.pop("E")
# 	F = li.pop("F")
# 	#Run model
# 	result = m(**li)
# 	print(result)
# 	sumE = result[0].sum()
# 	sumE.backward()
# 	print("Success!")
	
