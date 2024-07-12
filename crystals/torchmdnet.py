import warnings
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

import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots) #append top directory
from torchmdnet import torchmdnet
TorchMD_Net = torchmdnet.TorchMD_Net

import re, sys
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn, Tensor
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
# sys.path.append("torchmdnet")
# import torchmdnet
from torchmdnet import output_modules
from torchmdnet.wrappers import AtomFilter
from torchmdnet import priors
from torchmdnet.et import TorchMD_ET
from torchmdnet.utils import Distance, NeighborEmbedding, CosineCutoff
from crystals.general_train_eval import load_state

class Distance(Distance):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(Distance, self).__init__(cutoff_lower=cutoff_lower,
						cutoff_upper=cutoff_upper,
						max_num_neighbors=max_num_neighbors,
						return_vecs=return_vecs,
						loop=loop,)
        self.nbr_fea_to_vec = torch.nn.Linear(41, 3)

    def forward(self, pos:List[Tensor], batch):
#         edge_index = radius_graph(
#             pos,
#             r=self.cutoff_upper,
#             batch=batch,
#             loop=self.loop,
#             max_num_neighbors=self.max_num_neighbors,
#         )
#         edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        nbr_fea, dists, nbr_fea_idx = pos #(edges,41) and (edges,) and (2,edges)
        edge_weight = dists #(edges,)
        edge_index = nbr_fea_idx #(2,edges)
        edge_vec = self.nbr_fea_to_vec(nbr_fea) #NOT a true distance difference vector!, a PROXY for pymatgen format; (EDGES,3)
#         print(edge_vec)
	
        if self.loop:
            # mask out self loops when computing distances because
            # the norm of 0 produces NaN gradients
            # NOTE: might influence force predictions as self loop gradients are ignored
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        # TODO: return only `edge_index` and `edge_weight` once
        # Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        return edge_index, edge_weight, None

class NeighborEmbedding(NeighborEmbedding):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(hidden_channels=hidden_channels, num_rbf=num_rbf,
						cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, 
						max_z=max_z)
        self.embedding = torch.nn.Linear(92, hidden_channels)
        self.reset_parameters()

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        if False:	
            mask = edge_index[0] != edge_index[1]
            if not mask.all():
                edge_index = edge_index[:, mask]
                edge_weight = edge_weight[mask]
                edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1) #edges, hidden_c

        x_neighbors = self.embedding(z) #nodes, hidden_c
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class TorchMD_ET(TorchMD_ET):
    def __init__(
        self,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=5.0,
        max_z=100,
        max_num_neighbors=32,
    ):
        super(TorchMD_ET, self).__init__(hidden_channels=hidden_channels,
							num_layers=num_layers,
							num_rbf=num_rbf,
							rbf_type=rbf_type,
							trainable_rbf=trainable_rbf,
							activation=activation,
							attn_activation=attn_activation,
							neighbor_embedding=neighbor_embedding,
							num_heads=num_heads,
							distance_influence=distance_influence,
							cutoff_lower=cutoff_lower,
							cutoff_upper=cutoff_upper,
							max_z=max_z,
							max_num_neighbors=max_num_neighbors)
#         super().__init__()
	
        self.embedding = torch.nn.Linear(92, hidden_channels)
        self.distance = Distance(cutoff_lower, cutoff_upper, max_num_neighbors=max_num_neighbors, return_vecs=True, loop=False)
        self.expand_nbr_fea = torch.nn.Linear(41, max_num_neighbors)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z)
        self.reset_parameters()

    def forward(self,
                z: Tensor,
                pos: List[Tensor],
                batch: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        x = self.embedding(z) #input: nodes,92 -> output: nodes,hidden_c
        nbr_fea, dists, nbr_fea_idx = pos #(edges,41) and (edges,) and (2,edges)

        edge_index, edge_weight, edge_vec = self.distance(pos, batch)
#         print(edge_vec)
        assert (
            edge_vec is not None
        ), "Distance module did not return directional information"

        edge_attr = self.distance_expansion(edge_weight) #RBF!
	
        if False:
            mask = edge_index[0] != edge_index[1] #remove selfloop
            edge_vec[mask] = edge_vec[mask] / torch.norm(edge_vec[mask], dim=1).unsqueeze(1)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        vec = torch.zeros(x.size(0), 3, x.size(1), device=x.device)

        for attn in self.attention_layers:
            dx, dvec = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
#             print(x)
        x = self.out_norm(x)
#         print(x)

        return x, vec, z, pos, batch

class TorchMD_Net(TorchMD_Net):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
    ):
        super(TorchMD_Net, self).__init__(representation_model=representation_model,
						output_model=output_model,
						prior_model=prior_model,
						reduce_op=reduce_op,
						mean=mean,
						std=std,
						derivative=derivative)
        self.reset_parameters()

    def forward(self,
                atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch
                ) -> Tuple[Tensor, Optional[Tensor]]:

        # run the potentially wrapped representation model
        z = atom_fea #nodes, 92
        pos = (nbr_fea, dists, nbr_fea_idx) #(edges,41) and (edges,) and (2,edges)
	
        x, v, z, pos, batch = self.representation_model(z, pos, batch) #only place to worry about z and pos!
#         print(v)
        # apply the output network 
        x = self.output_model.pre_reduce(x, v, z, pos, batch) #no worries about pos and z!
#         print(x)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # apply output model after reduction
        out = self.output_model.post_reduce(out) #no worries about pos and z!

        return out

def create_model(args: dict, prior_model=None, mean=None, std=None):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "equivariant-transformer":
#         from et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
#     from wrappers import AtomFilter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(torchmdnet.priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    is_equivariant = False
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
    )
    return model
 

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

	config =  dict(activation= "silu",
			aggr= "add",
			atom_filter= -1,
			attn_activation= "silu",
			batch_size= 128,
			coord_files= "null",
			cutoff_lower= 0.0,
			cutoff_upper= 5.0,
			dataset= "QM9",
			dataset_arg= "energy_U0",
			dataset_root= "~/data",
			derivative= True,
			distance_influence= "both",
			distributed_backend= "ddp",
			early_stopping_patience= 150,
			ema_alpha_dy= 1.0,
			ema_alpha_y= 1.0,
			embed_files= "null",
			embedding_dimension= 256,
			energy_files= "null",
			energy_weight= 1.0,
			force_files= "null",
			force_weight= 1.0,
			inference_batch_size= 128,
			load_model= "null",
			log_dir= "logs/",
			lr= 0.0004,
			lr_factor= 0.8,
			lr_min= 1.0e-07,
			lr_patience= 15,
			lr_warmup_steps= 10000,
			max_num_neighbors= 64,
			max_z= 100,
			model= "equivariant-transformer",
			neighbor_embedding= True,
			ngpus= -1,
			num_epochs= 3000,
			num_heads= 8,
			num_layers= 8,
			num_nodes= 1,
			num_rbf= 64,
			num_workers= 6,
			output_model= "Scalar",
			precision= 32,
			prior_model= None,
			rbf_type= "expnorm",
			redirect= False,
			reduce_op= "add",
			save_interval= 10,
			splits= "null",
			standardize= False,
			test_interval= 10,
			test_size= "null",
			train_size= 110000,
			trainable_rbf= False,
			val_size= 10000,
			weight_decay= 0.0
		       )
	m = create_model(config)   
	path_and_name = os.path.join("/Scr/hyunpark/ArgonneGNN/argonne_gnn/save", "{}.pth".format("ctorchmdnet"))
	load_state(m, None, None, path_and_name, model_only=True)
	m.eval()

	#Get Dataloader
	import crystals.cgcnn_data_utils as cdu
	root_dir = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/CGCNN_test/data/imax"
	dataset = cdu.CIFData(root_dir)
	dataloader = cdu.get_dataloader(dataset, shuffle=False, **{'pin_memory': opt.pin_memory, 'persistent_workers': False,
                                 'batch_size': opt.batch_size})
	li = iter(dataloader).next()
	
	atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch, y = li.x, li.edge_attr, li.edge_index, li.edge_weight, li.cif_id, li.batch, li.y
	result = m(atom_fea, nbr_fea, nbr_fea_idx, dists, crystal_atom_idx, batch)
	print(result.view(-1), y)
	print(result.view(-1) - y)
	
# 	sumE = result.sum()
# 	sumE.backward()
# 	print("Success!")
