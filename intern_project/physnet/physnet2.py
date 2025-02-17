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
import os, glob, re
import curtsies.fmtfuncs as cf
import copy
import tqdm
import pickle
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter

class RBFLayer_Physnet(torch.nn.Module):
    def __init__(self, dfilter=100, cutoff=2, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('filter', torch.as_tensor(dfilter, dtype=torch.int32), persistent=False)
        self.register_buffer('cutoff', torch.as_tensor(cutoff, dtype=torch.float32), persistent=False)
        self.register_parameter("centers", torch.nn.Parameter(torch.nn.Softplus()(self._softplus_inverse(torch.linspace(1, torch.exp(-self.cutoff), self.filter.int().item()) )))) #(filter,)
        self.register_parameter("widths", torch.nn.Parameter(torch.nn.Softplus()(torch.as_tensor([self._softplus_inverse((0.5/((1.0-torch.exp(-self.cutoff))/self.filter))**2)] * self.filter.int().item() )))) #(filter,)

    def _softplus_inverse(self, x: torch.Tensor):
        """Numerically stable"""
        return x + torch.log(-torch.expm1(-x))

    def _cutoff_fn(self, dist_expanded):
        x = dist_expanded / self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return torch.where(x < 1, 1 - 6*x5 + 15*x4 - 10*x3, torch.zeros_like(x))

    def forward(self, edge_weight: torch.Tensor):
        assert edge_weight.ndim == 1, "Something has gone awry..." #Use graph noodes!
        edge_weight = edge_weight[:,None] #(nodes,1)
        cutoff_val = self._cutoff_fn(edge_weight) #(nodes,1)
        rbf = cutoff_val *  (-self.widths * (torch.exp(-edge_weight) - self.centers).pow(2)).exp() #(nodes, filter)
        return rbf #(nodes, filter)

class ResidualLayer_Physnet(torch.nn.Module):
    def __init__(self, filter0=256, filter1=256, activation_fn=None):
        super().__init__()
        self.activation = activation_fn
        self.dense = torch.nn.Linear(filter0, filter1)
        self.residual = torch.nn.Linear(filter1, filter1)
        self.dropout = torch.nn.Dropout()

    def forward(self, inputs: "Residual inputs"):
        if self.activation is not None: 
            h = self.dropout(self.activation(inputs))
        else:
            h = self.dropout(inputs)
        inputs = inputs + self.residual(self.dense(h))
        return inputs #(nodes, dim)

class InteractionLayer_Physnet(MessagePassing):
    def __init__(self, dfilter=100, filter=256, num_residuals=5, activation_fn=None):
        super().__init__(aggr="add") #MPNN method
        self.d2f = torch.nn.Linear(dfilter, filter)
        self.activation = activation_fn
        self.dense_i = torch.nn.Linear(filter, filter)
        self.residual_layers = torch.nn.ModuleList()
        for _ in range(num_residuals):
            self.residual_layers.append(ResidualLayer_Physnet(filter, filter, activation_fn))
        self.dense = torch.nn.Linear(filter, filter)
        self.register_parameter("u", torch.nn.Parameter(torch.ones([filter]), requires_grad=True))
        self.dropout = torch.nn.Dropout()

    def forward(self, h: torch.Tensor, edge_index: tuple, edge_weight: torch.Tensor, edge_attr: torch.Tensor, batch: torch.LongTensor):
        if self.activation is not None: 
            ha = self.dropout(self.activation(h))
        else:
            ha = self.dropout(h)
			
        row, col = edge_index
        g = self.d2f(edge_attr) #(edges, dim)
         
        #calculate contribution of neighbors and central atom
        y = self.dense_i(ha) #(nodes, dim)

        # element-wise multiplication, aggregating and Dense layer
# 	y = torch.index_select(y, 0, col) #(edges, dim)
#         y = y * g #(edges, dim)
#         m = scatter(y, row, dim=0, reduce=self.readout) #(nodes, dim)
        m = self.propagate(edge_index, y=y, g=g) #Above three lines in one

	#add contributions to get the "message" 
        for i, residual_layer in enumerate(self.residual_layers):
            m = residual_layer(m)
        if self.activation is not None: 
            m = self.activation(m)
        x = self.u * h + self.dense(m) #(nodes, dim)
        return x, g #x: (nodes, dim); g: (edges, dim)

    def message(self, y_j, g):
        return y_j * g #(edges, dim)

class InteractionBlock_Physnet(torch.nn.Module):
    def __init__(self, dfilter=100, filter=256, num_residuals=5, num_residuals_atomic=5, activation_fn=None):
        super().__init__()
        self.interactions = InteractionLayer_Physnet(dfilter=dfilter, filter=filter, num_residuals=num_residuals, activation_fn=activation_fn)
        self.residual_layers = torch.nn.ModuleList()
        for _ in range(num_residuals_atomic):
            self.residual_layers.append(ResidualLayer_Physnet(filter, filter, activation_fn))

    def forward(self, h: torch.Tensor, edge_index: tuple, edge_weight: torch.Tensor, edge_attr: torch.Tensor, batch: torch.LongTensor):
        h, g = self.interactions(h, edge_index, edge_weight, edge_attr, batch)
        for i, residual_layer in enumerate(self.residual_layers):
            h = residual_layer(h)
        return h, g

class OutputBlock_Physnet(torch.nn.Module):
    def __init__(self, dfilter=100, filter=256, num_residuals=5, num_residuals_atomic=3, activation_fn=None):
        super().__init__()
        self.residual_layers = torch.nn.ModuleList()
        self.activation_fn = activation_fn
        for _ in range(num_residuals_atomic):
            self.residual_layers.append(ResidualLayer_Physnet(filter, filter, activation_fn))
        self.dense = torch.nn.Linear(filter, 1)

    def forward(self, h: torch.Tensor, edge_index: tuple, edge_weight: torch.Tensor, edge_attr: torch.Tensor, batch: torch.LongTensor):
        for i, residual_layer in enumerate(self.residual_layers):
            h = residual_layer(h)
        if self.activation_fn is not None: 
            h = self.activation_fn(h)
        h = self.dense(h)
        return h

class Physnet(torch.nn.Module):
    def __init__(self, dfilter=100, filter:int=256, cutoff=12, num_residuals=2,
		 num_residuals_atomic=2, num_interactions=2, num_outer_residuals=1, activation_fn=torch.nn.ReLU(), 
		 dmodel:int=20, token_embedding_necessary=True, 
		 max_num_neighbors=32, **kwargs):
        super().__init__()  
        self.embedding = torch.nn.Embedding(100, filter) #100 elements by filter size
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.rbf = RBFLayer_Physnet(dfilter=dfilter, cutoff=cutoff)
        self.interactions = torch.nn.ModuleList([InteractionBlock_Physnet(dfilter=dfilter, filter=filter, num_residuals=num_residuals, num_residuals_atomic=num_residuals_atomic, activation_fn=activation_fn) for _ in range(num_interactions)])
        self.output_blocks = torch.nn.ModuleList([OutputBlock_Physnet(dfilter=dfilter, filter=filter, num_residuals=num_residuals, num_residuals_atomic=num_residuals_atomic, activation_fn=activation_fn) for _ in range(num_interactions)])
        self.readout = kwargs.get("readout", "sum")
        self.explain = kwargs.get("explain", False)
        std = kwargs.get("std", None)
        mean = kwargs.get("mean", None)
        self.std = std
        self.mean = mean
        if self.explain:
            def hook(module, inputs, grad):
                self.embedded_grad = grad
            self.embedding.register_backward_hook(hook)
	
        self.reset_all_weights()
	
    def reset_all_weights(self, ) -> None:
        """
        refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        """

        @torch.no_grad()
        def weight_reset(m: torch.nn.Module):
             # - check if the current module has reset_parameters & if it's callabed called it on m
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        self.apply(fn=weight_reset)
	
    def _init_weights(self, m: torch.nn.Module):
        if isinstance(m, torch.nn.Linear):
            if hasattr(m, 'weight'): torch.nn.init.xavier_normal_(m.weight)
            elif hasattr(m, 'bias'):  torch.nn.init.constant_(m.bias)
        else:
            if hasattr(m, 'weight'): torch.nn.init.normal_(m.weight)
            # elif hasattr(m, 'bias'): torch.nn.init.constant_(m.bias, 0)
                 
    def forward(self, z, pos, batch: torch.LongTensor=None, metadata: dict=None):
        #pos #(nodes, 3)
        #z #(nodes,)
        pos.requires_grad_()

        h = self.embedding(z) #(nodes, dim)

        edge_index = metadata["edge_index"] if metadata != None and metadata["edge_index"] else radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1) #distance (nodes,)
        edge_attr = self.rbf(edge_weight) #(edges, dfilter)
		
        E = 0.
        for i, (interaction, outmodule) in enumerate(zip(self.interactions, self.output_blocks)):
            h, _ = interaction(h, edge_index, edge_weight, edge_attr, batch) #x: (nodes, dim); g: (edges, dim)
            Eh = outmodule(h, edge_index, edge_weight, edge_attr, batch) #h: (nodes, dim) -> (nodes, dim=1)
            E += Eh
        x = E #(nodes, dim=1)
	
        if self.explain:
            self.final_conv_acts = h #Last interaction layer (nodes, dim)
            def hook(grad):
                self.final_conv_grads = grad
            self.final_conv_acts.register_hook(hook) #only when backpropped!	
        
        if self.mean is not None and self.std is not None:
#             print("Physnet", self.std, self.mean)
#             x = x * self.std + self.mean
            pass
            
        out = scatter(x, batch, dim=0, reduce=self.readout) #(batch, 1)	
		
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
        dy = torch.autograd.grad(
            [out],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        if dy is None:
            raise RuntimeError("Autograd returned None for the force prediction.")
#         return dict(E=out, F=-dy)
        return out, -dy

if __name__=="__main__":
    physnet_block={'dfilter':64, 
			   'filter':64, 
			   'cutoff':10, 
			   'num_residuals':3, 
			   'num_residuals_atomic':2, 
			   'num_interactions':5, 
			   'activation_fn':torch.nn.SiLU(), 
			   'dmodel': 64,
			   "token_embedding_necessary": True,
			   "threshold":5.0,
			   "bond": bonds_info,
     			   "coord_ref": coord_ref_info,
                           "species": species_info,
                           "bond_energy_add": False}
    model = Physnet(**physnet_block) 
    model.cuda()
    protein_system = "alanine_dipeptide"
    trainloaded, validloaded, bonds, coord_ref = dl.DataLoader(batch_size=4000, protein_name=protein_system).dataloader #8000 for decaalanine; 2000 for pentapeptide
    coords, species = next(iter(trainloaded))
    coords, species = coords.cuda(), species.cuda()
    coord_ref = coord_ref[0]
    coord_ref = coord_ref.cuda()

    optim = torch.optim.Adam(model.parameters())
    E = model([species, coords, bonds, coord_ref])
    optim.zero_grad()
    loss = E[0].mean()
    loss.backward()
    optim.step()
    print("Done")
