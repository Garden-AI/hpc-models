"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
import os, glob, re, sys, gc
from torch_geometric.nn import MessagePassing, radius_graph
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory
from typing import Tuple, Union, Optional
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.utils import BaseSettings
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
import curtsies.fmtfuncs as cf
import copy
import tqdm
import pickle
from torch_scatter import scatter
from alignn import alignn
from alignn.alignn import compute_bond_cosines, ALIGNNConfig

ALIGNN = alignn.ALIGNN
import warnings
warnings.simplefilter("ignore")

class ALIGNN(ALIGNN):
    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__(config=config)
        self.atom_embedding = torch.nn.Linear(92, config.hidden_features)

    def forward(
        self, atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch, metadata=None):
        import faulthandler
        faulthandler.enable()
        device = torch.cuda.current_device()

        g = self.geometric_to_dgl_format(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch, cutoff=self.cutoff, 
                                         max_num_neighbors=self.max_num_neighbors,
                                         metadata=metadata, node_feat_dim=self.node_features) #batched graphs!!!
        g = g.to(device)	
        g = g.local_var()
        # initial node features: atom feature network...
        
        x = g.ndata.pop("atom_features") #Already embedded
        x = self.atom_embedding(x)

        # initial bond features
        row, col = g.edges()
        r = g.edata["r"] #(edges,3)
        bondlength = g.edata["dists"] #(edges,)
        y = self.edge_embedding(bondlength)

        if len(self.alignn_layers) > 0:
            lg = g.line_graph(shared=True)
            lg.apply_edges(compute_bond_cosines)
#             lg = lg.local_var()
            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
        
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
            
        return out  

    @staticmethod
    def geometric_to_dgl_format(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch, cutoff=10, max_num_neighbors=32, metadata: dict=None, node_feat_dim=100):
        atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch = list(map(lambda inp: inp.to("cpu"), [atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch] ))
        unique_batches = batch.unique() #LongTensor unique
        import collections
#         print(collections.Counter(batch.detach().cpu().numpy().tolist()))
#         print(nbr_fea_idx.shape)
        bg = []
        start_atom_idx = 0
        
        for ub in unique_batches:            
            idx = ub == batch
            batch_ = batch[idx]
            atom_fea_ = atom_fea[idx]
#             print(ub, len(atom_fea_), sum(idx))
            
            end_atom_idx = len(atom_fea_) + start_atom_idx
            row_only = nbr_fea_idx[0] #for row vector
            row_only_idx = torch.logical_and((row_only >= start_atom_idx), (row_only < end_atom_idx))
#             print(row_only_idx)
            row, col = nbr_fea_idx[:,row_only_idx]
#             print(row,col, start_atom_idx)
            row -= start_atom_idx #For each graph, index starts from 0!
            col -= start_atom_idx
#             print(row,col)
            g = dgl.graph((row, col))
            start_atom_idx = end_atom_idx

            if metadata != None and metadata["atom_features"]:
                v = metadata["atom_features"][idx] #Special case for ALIGNN (nodes, node_feats)
                g.ndata["atom_features"] = v #(nodes, node_feats)
            else:
#                 print(atom_fea_.shape)
                g.ndata["atom_features"] = atom_fea_
            bg.append(g)
			
        bg = dgl.batch(bg)
        bg.edata["dists"] = dists
        bg.edata["r"] = nbr_fea

        del atom_fea
        del nbr_fea
        del nbr_fea_idx
        del dists
        del crystal_atom_idx
        del batch
        gc.collect()

        return bg #batched dgl graphs
    
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
	    parser.add_argument('--dataset', type=str, default="cifdata", choices=["qm9", "md17", "ani1", "ani1x", "qm9edge", "moleculenet","cgcnn","cifdata","gandata","cdvaedata"])
	    parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/data")
	    parser.add_argument('--data_dir_crystal', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax")
	    parser.add_argument('--task', type=str, default="homo")
	    parser.add_argument('--pin_memory', type=bool, default=True) #causes CUDAMemory error;; asynchronously reported at some other API call
	    parser.add_argument('--use_artifacts', action="store_true", help="use artifacts for resuming to train")
	    parser.add_argument('--use_tensors', action="store_true") #for data, use DGL or PyG formats?
	    parser.add_argument('--crystal', action="store_true") #for data, use DGL or PyG formats?
	    parser.add_argument('--make_data', action="store_true", help="force making data")

	    # train
	    parser.add_argument('--epoches', type=int, default=2)
	    parser.add_argument('--batch_size', type=int, default=4) #Per GPU batch size
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
	    parser.add_argument('--backbone', type=str, default='physnet', choices=["schnet","physnet","torchmdnet","alignn","dimenet","dimenetpp","cgcnn","cphysnet","cschnet","ctorchmdnet"])
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
	config = alignn.ALIGNNConfig(name="alignn")
	m = ALIGNN(config)
    
   	 #Get Dataloader
	import crystals.dataloader as dl
	root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax"
	dataloader = dl.DataModuleCrystal(opt=opt).train_dataloader()
	li = iter(dataloader).next()
	
	atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, batch, dists, y = li.x, li.edge_attr, li.edge_index, li.cif_id, li.batch, li.edge_weight, li.y
	result = m(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch)
	
	print(result.view(-1), y)
	print(result.view(-1) - y)
	(result.view(-1,)-y).sum().backward()
	print("Good")
    
