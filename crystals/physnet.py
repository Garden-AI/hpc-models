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
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory
from physnet import physnet
Physnet = physnet.Physnet
import warnings
warnings.simplefilter("ignore")

class Physnet(Physnet):
    def __init__(self, dfilter=100, filter:int=256, cutoff=12, num_residuals=2,
		 num_residuals_atomic=2, num_interactions=2, activation_fn=torch.nn.ReLU(), 
		 dmodel:int=20, token_embedding_necessary=True, 
		 max_num_neighbors=32, **kwargs):
        super().__init__(dfilter=dfilter, filter=filter, cutoff=cutoff, num_residuals=num_residuals,
		 num_residuals_atomic=num_residuals_atomic, num_interactions=num_interactions, activation_fn=activation_fn, 
		 dmodel=dmodel, token_embedding_necessary=token_embedding_necessary, 
		 max_num_neighbors=max_num_neighbors, **kwargs)  
        self.embedding = torch.nn.Linear(92,filter)
        self.expand_nbr_fea = torch.nn.Linear(41, dfilter)
        self.readout = "mean"
        self.apply(self._init_weights)
        self.return_property = kwargs.get("return_property", True)

        if self.explain:
            def hook(module, inputs, grad):
                self.embedded_grad = grad
            self.embedding.register_backward_hook(hook)
			
    def _init_weights(self, m: torch.nn.Module):
        super()._init_weights(m=m)
                 
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch=None, metadata: dict=None):
        #pos #(nodes, 3)
        #z #(nodes,)
#         pos.requires_grad_()

        h = self.embedding(atom_fea) #(nodes, 92) -> (nodes,filter)

        edge_weight = dists
        edge_attr = self.expand_nbr_fea(nbr_fea) #(edges, dfilter)
		
        for i, interaction in enumerate(self.interactions):
            h, _ = interaction(h, nbr_fea_idx, edge_weight, edge_attr, batch) #x: (nodes, dim); g: (edges, dim)
	
        x = h
        for dense in self.dense_post_list:
            x = dense(x) #num_nodes, 100
	
        if self.explain:
            self.final_conv_acts = x
            def hook(grad):
                self.final_conv_grads = grad
            self.final_conv_acts.register_hook(hook) #only when backpropped!	
        
        if self.return_property:
            x = self.last(x) #num_nodes, 1
            out = scatter(x, batch, dim=0, reduce=self.readout) #(batch, 1)	
        else:
            out = x #num_nodes, 100
			
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

	#Initialize model
	config = dict(dfilter=64, 
						 filter=64, 
						 cutoff=10, 
						 num_residuals=3,
						 num_residuals_atomic=2, 
						 num_interactions=5, 
						 activation_fn=torch.nn.ReLU(), 
						 dmodel=64, 
						 token_embedding_necessary=True, 
						 max_num_neighbors=32,
						 readout="sum")
	m = Physnet(**config)
	from crystals.general_train_eval import load_state
	path_and_name = os.path.join("/Scr/hyunpark/ArgonneGNN/argonne_gnn/save", "{}.pth".format("cphysnet"))
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

# 	sumE = result.sum()
# 	sumE.backward()
# 	print("Success!")

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
	
