import torch
import torch.nn as nn
from datetime import datetime
from matplotlib import pyplot as plt
import functools
import pandas as pd
from sklearn import preprocessing
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import itertools, functools, operator, collections
from typing import *
import shutil
import networkx as nx
import einops
from einops.layers.torch import Rearrange, Reduce
import warnings
import os, glob, re, math, sys
import curtsies.fmtfuncs as cf
import copy
import tqdm
import pickle
from torch_geometric.nn import MessagePassing, radius_graph, GATv2Conv, GATConv
from torch_scatter import scatter
from transformers import AdamW
import argparse
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots)
from crystals.dataloader_mof import *
from torch_geometric.loader import DataLoader #Can this handle DDP? yeah!
from torch.nn.utils.parametrizations import spectral_norm
import torch.distributed as dist
from curtsies import fmtfuncs as cf
from train.dist_utils import to_cuda, get_local_rank, init_distributed, seed_everything, \
    using_tensor_cores, increase_l2_fetch_granularity, Logger, WandbLogger
from train.gpu_affinity import set_affinity
from torch.nn.parallel import DistributedDataParallel
from p_tqdm import *
import wandb 
from crystals.atomic_properties import *

#POLYNOMIAL GAN https://github.com/grigorisg9gr/polynomial_nets/blob/4fdb13c79d95b99e7adcbb369d6b33213f771fa2/image_generation_chainer/gen_models/cnn_gen_custom_prodpoly.py#L252:~:text=class%20ProdPolyConvGenerator(chainer.Chain)%3A

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']
short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']

class LinearModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid=False, use_attention=True):
        super().__init__()
        self.sigmoid = sigmoid
        self.use_attention = use_attention
        if not use_attention:
            if sigmoid:
                self.add_module("linear_module", torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels), 
                                                               torch.nn.Sigmoid()))
            else:
                self.add_module("linear_module", torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels), 
                                                               torch.nn.SiLU(inplace=True)))
        else:
            if sigmoid:
                self.add_module("linear_module", torch.nn.Sequential(GATv2Conv(in_channels, out_channels), 
                                                               torch.nn.Sigmoid()))
            else:
                self.add_module("linear_module", torch.nn.Sequential(GATv2Conv(in_channels, out_channels), 
                                                   torch.nn.SiLU(inplace=True)))

    def forward(self, inputs: List[torch.Tensor]):
        input_tensor, edge_index = inputs
#         print(self.linear_module[1](self.linear_module[0](input_tensor, edge_index)))
        
#         return self.linear_module(input_tensor, edge_index)
        if self.use_attention:
            return self.linear_module[1](self.linear_module[0](input_tensor, edge_index))
        else:
            return self.linear_module[1](self.linear_module[0](input_tensor))

class Element(torch.nn.Module):
    def __init__(self, atom_embedding: torch.Tensor=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 8) if atom_embedding == None else atom_embedding
        self.linear = LinearModule(8, 128) #3 atoms

    def forward(self, inputs: List[torch.Tensor]):
        input_tensor, edge_index = inputs
        x = input_tensor.long().view(-1,) #Numatoms
        b = x.size() #Numatoms
        x = self.embedding.index_select(index=x, dim=0).view(-1, 8) if atom_embedding != None else self.embedding(x) #(Numatoms, 23) 
#         print(x.shape, edge_index.shape)
        x = self.linear([x, edge_index]) #Numatoms, 128
#         print("el", x)
        return x

class Latent(torch.nn.Module):
    def __init__(self, lat_dim: int=100):
        super().__init__()
        self.linear = LinearModule(lat_dim, 256)

    def forward(self, inputs: List[torch.Tensor]):
        input_tensor, edge_index = inputs
        x = self.linear([input_tensor, edge_index]) #-> NumMOFs, 256
        return x
    
class Discriminator(torch.nn.Module):
    def __init__(self, linear_block_channels=[18, 128, 256, 256, 128, 64, 1]):
        super().__init__()
        
        linear_list = torch.nn.ModuleList([])
        input_cs = linear_block_channels[:-1]
        output_cs = linear_block_channels[1:]
        for inc, outc in zip(input_cs, output_cs):
            linear_list.append(LinearModule(inc, outc, use_attention=False))
        self.add_module("linear_block", torch.nn.ModuleList(linear_list))

        self.add_module("final_linear", torch.nn.Sequential(torch.nn.Linear(output_cs[-1], output_cs[-1]), torch.nn.SiLU(True), 
                                                            torch.nn.Linear(output_cs[-1], output_cs[-1]), torch.nn.SiLU(True), 
                                                            torch.nn.Linear(output_cs[-1], 1)) )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, inputs: List[torch.Tensor]):
        tensor_input, batch, edge_idx = inputs
        numatoms, c = tensor_input.size()
        x = tensor_input
        for module in self.linear_block:
            x = module([x, edge_idx]) #(Numatoms,dim)
        x = scatter(src = x, index = batch, dim=0, reduce="mean") #(NumMOFs, dim)
        x = self.final_linear(x) # -> (NumMOFs, 1)
        return x #(NumMOFs, 1)

class Generator(torch.nn.Module):
    def __init__(self, linear_block_channels=[256, 128, 64, 32, 16, 3], atom_embedding: torch.Tensor=None, lat_dim: int=None):
        super().__init__()
        
        self.add_module("element", Element(atom_embedding))
        
        self.add_module("latent", Latent(lat_dim))

        self.add_module("pre_linear", LinearModule(384, linear_block_channels[0]))
        
        linear_list = torch.nn.ModuleList([])
        input_cs = linear_block_channels[:-1]
        output_cs = linear_block_channels[1:]
        for idx, (inc, outc) in enumerate(zip(input_cs, output_cs)):
            if idx != (len(input_cs) - 1):
                linear_list.append(LinearModule(inc, outc))
            else:
                linear_list.append(LinearModule(inc, outc, sigmoid=True, use_attention=True))
        self.add_module("linear_block_for_coords", torch.nn.ModuleList(linear_list))

        self.add_module("post_linear_for_lengths", LinearModule(output_cs[-1], 3))
        self.add_module("post_linear_for_angles", LinearModule(output_cs[-1], 3))

        self.add_module("linear_block_for_lengths", torch.nn.ModuleList([LinearModule(3, 3) for _ in range(4)]) )
        self.add_module("linear_block_for_angles", torch.nn.ModuleList([LinearModule(3, 3) for _ in range(4)]) )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01)
            
    def forward(self, inputs: List[torch.Tensor]):
        el, lat, batch, edge_idx = inputs
#         print(el.shape, lat.shape, batch.shape, edge_idx.shape)
        el, lat = list(map(lambda func, inp, edge: func([inp, edge]), (self.element, self.latent), (el, lat), (edge_idx, edge_idx) )) #(Numatoms, 128) and (Numatoms, 256)

        x = torch.cat([el,lat], dim=-1) #(Numatoms, 384)
        x = self.pre_linear([x, edge_idx]) #(Numatoms, 384) -> #(Numatoms, 256)
        
        for module in self.linear_block_for_coords:
            x = module([x, edge_idx]) #(Numatoms, 256) -> (Numatoms, 3)
        coords = x

        lengths = self.post_linear_for_lengths([coords, edge_idx]) #(Numatoms, 3)        
#         lengths = scatter(src = lengths, index = batch, dim=0) #(Numatoms, 3) -> (NumMOFs, 3)
        for module in self.linear_block_for_lengths:
            lengths = module([lengths, edge_idx]) #(Numatoms, 3) -> (Numatoms, 3)
        lengths = scatter(src = lengths, index = batch, dim=0, reduce="mean") #(Numatoms, 3) -> (NumMOFs, 3)
        
        angles = self.post_linear_for_angles([coords, edge_idx]) #(Numatoms, 3)
        for module in self.linear_block_for_angles:
            angles = module([angles, edge_idx])  #(Numatoms, 3) -> (Numatoms, 3)
        angles = scatter(src = angles, index = batch, dim=0, reduce="mean") #(Numatoms, 3) -> (NumMOFs, 3)
#         print(coords, lengths, angles)
        return coords, lengths, angles #(Numatoms, 3), (NumMOFs, 3), (NumMOFs, 3)

class GAN(torch.nn.Module):
    def __init__(self, g_linear_block_channels=[256, 128, 64, 32, 16, 3], atom_embedding: torch.Tensor=None, lat_dim: int=None,
                                        d_linear_block_channels=[18, 128, 256, 256, 128, 64, 1]):
        super().__init__()
        self.generator = Generator(g_linear_block_channels, atom_embedding, lat_dim)
        self.discriminator = Discriminator(d_linear_block_channels)
        
    def forward(self, x):
        return x
    
def calc_grad_penalty_cubic(netD, device, real_samples, fake_samples, batch_real, edge_index, FLAGS):
    """ Calculates the gradient penalty.
    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
    """
    #-------------
    #credits to https://keras.io/examples/generative/wgan_gp/
    #-------------
    # get the interplated image
#     alpha = torch.randn(FLAGS.sample_size, 1).to(real_samples)
    alpha = torch.rand(real_samples.size(0), 1).to(real_samples) #(Natoms, 1)
#     diff = fake_samples - real_samples
    interpolated = alpha * real_samples.data + (1 - alpha) * fake_samples.data #just a leaf-tensor
    
    interpolated.requires_grad_(True)
    netD.train()
    [p.requires_grad_(False) for p in netD.parameters()]
    pred = netD([interpolated, batch_real, edge_index])
    [p.requires_grad_(True) for p in netD.parameters()]

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = torch.autograd.grad(pred, interpolated, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    grads = grads.view(FLAGS.sample_size, -1)
    grad_norm_sqrt = grads.norm(dim=-1)
    gp = (grad_norm_sqrt - 1.0).pow(2).mean()
    return gp
    

# def generate_real_samples(dataset, n_samples, n_element,n_spacegroup, ix):
def generate_real_samples(dataset, n_samples, ix):
#     X_node_class = torch.cat([dataset[i].node_class for i in ix], dim=0).long() #(NumAtoms,*)
#     X_node_target = torch.cat([dataset[i].node_target for i in ix], dim=0).long() #(NumAtoms,*)
#     X_node_simple_feature = torch.cat([dataset[i].node_simple_feature for i in ix], dim=0).float() #(NumAtoms,*)
#     X_node_radial_feature = torch.cat([dataset[i].node_radial_feature for i in ix], dim=0).float() #(NumAtoms,*)
#     X_coords = torch.cat([dataset[i].node_radial_feature for i in ix], dim=0).float() #(NumAtoms,*)
#     X_lengths = torch.stack([dataset[i].lengths for i in ix], dim=0).float() #(NumMOF,*)
#     X_angles = torch.stack([dataset[i].angles for i in ix], dim=0).float() #(NumMOF,*)
    X_node_class = dataset.node_class
    X_node_target = dataset.node_target
    X_node_simple_feature = dataset.node_simple_feature
    X_node_radial_feature = dataset.node_radial_feature 
    X_coords = dataset.coords
    X_lengths = dataset.lengths
    X_angles = dataset.angles
    batches = dataset.batch #(NumAtoms,*)
    edges = dataset.edge_index
    return X_node_class, X_node_target, X_node_simple_feature, X_node_radial_feature, X_coords, X_lengths, X_angles, batches, edges
    
def generate_fake_lables(n_samples, aux_data: "List of 1. elements 2. num of elements to generate"):
#     label_sp = np.random.choice(aux_data[1], n_samples, p=aux_data[-1]).reshape(-1,1)
    label_elements = [] #Natoms,
    batch = [] #Natoms,
    list_of_elements = aux_data[0]
    el2num = {key: idx for idx, key in enumerate(list_of_elements)}
    list_of_elements: List[int] = list(el2num.values())
    
    for idx, i in enumerate(range(n_samples)):
        num_atoms_to_gen: int = aux_data[1][i].item()
        fff = np.random.choice(list_of_elements, num_atoms_to_gen, replace=False)
        label_elements.extend(fff.tolist())
        batch.extend([idx] * num_atoms_to_gen)
    label_elements = np.array(label_elements) #(Numatoms,)
    batch = np.array(batch)
    return list(map(lambda inp: torch.from_numpy(inp).type(torch.long), [label_elements, batch] )) #(Numatoms, )

# Define the loss functions to be used for discriminator
# This should be (fake_loss - real_loss)
def discriminator_loss(real_spl, fake_spl):
    real_loss = torch.mean(real_spl)
    fake_loss = torch.mean(fake_spl)
    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_spl):
    return -torch.mean(fake_spl)

def train_step(X_real: List[torch.Tensor], device, model_path_and_name, gan_model, atom_embedding, FLAGS, n_discriminator, d_optimizer, g_optimizer):
    d_model = gan_model.discriminator
    g_model = gan_model.generator
    
    #symmetry,elements,coords,lengths,angles
    real_crystal = [tensor.to(device) for tensor in X_real] #What about device ?
#     print(real_crystal)
    #real_crystal = [X_node_class, X_node_target, X_node_simple_feature, X_node_radial_feature, X_coords, X_lengths, X_angles, batches]

    ####***BASIC***####
    batch_real = real_crystal[7] #(Numatoms,)
    edge_index = real_crystal[8] #(2, Numedges)
    _, num_atoms_to_gen = batch_real.unique(return_counts=True) #(NumMOFs,)
    elem = real_crystal[0].view(-1,1) #(Numatoms,1)
#     e = torch.cat([real_crystal[2], real_crystal[3]], dim=-1) #WIP: Use it or NOT? (Numatoms, dim2+dim3); embedding
    e=atom_embedding.index_select(index=elem.view(-1,), dim=0).view(-1, 8) #(Numatoms, dim)
    coords = real_crystal[4] #(Numatoms, 3)
    lengths = real_crystal[5].float()
    angles = real_crystal[6].float()
    lengths = torch.repeat_interleave(lengths, num_atoms_to_gen, dim=0, output_size=num_atoms_to_gen.sum().item()) #(NumMOFs, 3) -> (Numatoms, 3) #Already done?
    angles = torch.repeat_interleave(angles, num_atoms_to_gen, dim=0, output_size=num_atoms_to_gen.sum().item()) #(NumMOFs, 3) -> (Numatoms, 3)
    list_of_elements = list(en_pauling.keys())
    ####***BASIC***####
    
    real_crystal = torch.cat((e, coords, lengths, angles), dim=-1).to(device) #Numatoms, DIM=(1+8+3+3+3)=18

    original_sample_size = FLAGS.sample_size
    FLAGS.sample_size = num_atoms_to_gen.size(0) #How many CIFs (NumMOFs)?
    AUX_DATA = [list_of_elements, num_atoms_to_gen]
    
    for _ in range(n_discriminator):
        d_model.train()
        g_model.eval()
        
        ####***Generator Production***####
        noise = torch.randn(FLAGS.sample_size, FLAGS.lat_dim).to(device) #(NumMOFs, lat_dim) ;;Device?
        fake_labels, _ = generate_fake_lables(FLAGS.sample_size, AUX_DATA) #batch_fake = batch_real
        _, num_atoms_to_gen = batch_real.unique(return_counts=True) #batch_fake = batch_real
        noise = torch.repeat_interleave(noise, num_atoms_to_gen, dim=0) #(NumMOFs, 256) -> (NumAtoms, 256)
        elem = fake_labels.view(-1,1) #(Numatoms, ) -> (Numatoms, 1)
        fake_crystal = g_model([elem.to(device), noise, batch_real, edge_index]) #batch_fake = batch_real
        ####***Generator Production***####

        ####***Generator BASIC***####
        e=atom_embedding.index_select(index=elem.view(-1,), dim=0).view(-1, 8) #(Numatoms, dim)
        coords = fake_crystal[0].to(device) #(Numatoms, 3)
        lengths = torch.repeat_interleave(fake_crystal[1], num_atoms_to_gen, dim=0) #(NumMOFs, 3) -> (Numatoms, 3)
        angles = torch.repeat_interleave(fake_crystal[2], num_atoms_to_gen, dim=0) #WIP for angles: (NumMOFs, 3) -> (Numatoms, 3)        
        fake_crystal = torch.cat((e, coords, lengths, angles), dim=-1).to(device) #Numatoms, DIM=(1+8+3+3+3)=18
        ####***Generator BASIC***####

        ####***Discriminator TRAINING***####
        fake_logits, real_logits = d_model([fake_crystal, batch_real, edge_index]), d_model([real_crystal, batch_real, edge_index]) #(NumMOFs,1) #batch_fake = batch_real
        d_loss = discriminator_loss(real_logits, fake_logits)
        gp = calc_grad_penalty_cubic(d_model, device, real_crystal, fake_crystal, batch_real, edge_index, FLAGS) #compare concats
#         print(fake_crystal-real_crystal)
        d_loss = d_loss + 10.0*gp
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        ####***Discriminator TRAINING***####

    d_model.eval()
    g_model.train()
    
    ####***Generator TRAINING***####
    noise = torch.randn(FLAGS.sample_size, FLAGS.lat_dim).to(device) #(NumMOFs, lat_dim) ;;Device?
    fake_labels, _ = generate_fake_lables(FLAGS.sample_size, AUX_DATA)
    _, num_atoms_to_gen = batch_real.unique(return_counts=True) #batch_fake = batch_real
    noise = torch.repeat_interleave(noise, num_atoms_to_gen, dim=0) #(NumMOFs, 256) -> (NumAtoms, 256)
    elem = fake_labels.view(-1,1) #(Numatoms, ) -> (Numatoms, 1)
    fake_crystal = g_model([elem.to(device), noise, batch_real, edge_index])
    
    #checkpoint
    e=atom_embedding.index_select(index=elem.view(-1,), dim=0).view(-1, 8) #(Numatoms, dim)
    coords = fake_crystal[0].to(device) #(Numatoms, 3)
    lengths = torch.repeat_interleave(fake_crystal[1], num_atoms_to_gen, dim=0) #(NumMOFs, 3) -> (Numatoms, 3)
    angles = torch.repeat_interleave(fake_crystal[2], num_atoms_to_gen, dim=0) #WIP for angles: (NumMOFs, 3) -> (Numatoms, 3)        
    fake_crystal = torch.cat((e, coords, lengths, angles), dim=-1).to(device) #Numatoms, DIM=(1+8+3+3+3)=18
    
    #checkpoint
    gen_crystal_logits = d_model([fake_crystal, batch_real, edge_index])
    g_loss = generator_loss(gen_crystal_logits)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
    ####***Generator TRAINING***####

    FLAGS.sample_size = original_sample_size
    return d_loss.item(), g_loss.item()

def save_states(gan_model, g_optimizer, d_optimizer, epoch, loss, model_path_and_name):
    if get_local_rank() == 0:
        if not isinstance(gan_model, torch.nn.parallel.DistributedDataParallel):
            states = dict(gan_model=gan_model.state_dict(), g_optimizer=g_optimizer.state_dict(), d_optimizer=d_optimizer.state_dict(), epoch=epoch, loss=loss)
        else:
            states = dict(gan_model=gan_model.module.state_dict(), g_optimizer=g_optimizer.state_dict(), d_optimizer=d_optimizer.state_dict(), epoch=epoch, loss=loss)
        torch.save(states, f"{model_path_and_name}")
        print("Saving a GAN model...")
    
def load_states(gan_model, g_optimizer, d_optimizer, model_path_and_name, device, FLAGS, logger):
    
#     model_name = os.path.split(model_path_and_name)[0].split("/")[-1] #gan.pt
#     name = model_name.split(".")[0] #gan
    name = os.path.splitext(FLAGS.model_filename)[0] #gan
    if get_local_rank() == 0 and FLAGS.use_artifacts:
#         print(model_path_and_name, model_name, name) #WIP: name is wrong!
        prefix_dir = logger.download_artifacts(f"{name}_model_objects")
        shutil.copy(os.path.join(prefix_dir, name + ".pt"), model_path_and_name + ".artifacts") 
        model_path_and_name += ".artifacts"
    
    ckpt = torch.load(model_path_and_name, map_location={'cuda:0': f'cuda:{get_local_rank()}'})
    (gan_model.load_state_dict(ckpt["gan_model"]) if not isinstance(gan_model, torch.nn.parallel.DistributedDataParallel) else gan_model.module.load_state_dict(ckpt["gan_model"]))
    g_optimizer.load_state_dict(ckpt["g_optimizer"])
    d_optimizer.load_state_dict(ckpt["d_optimizer"])
    epoch = ckpt["epoch"]
    loss = ckpt["loss"]
    return epoch, loss

def plot_history(d_hist, g_hist):
    plt.plot(d_hist)
    plt.xlabel('step (s)')
    plt.savefig('logs/d_loss-%d.png'%(FLAGS.device))
    plt.close()

    plt.plot(g_hist)
    plt.xlabel('step (s)')
    plt.savefig('logs/g_loss-%d.png'%(FLAGS.device))


def generate_latent_inputs(lat_dim,n_samples,candidate_element_comb,aux_data,FLAGS):
    z = np.random.normal(0,1.0,(n_samples, lat_dim))
    p = aux_data[-1]
    label_sp = np.random.choice(aux_data[1],n_samples,p=p).reshape(-1,1)

    with open('cubic-elements-dict.json', 'r') as f:
        e_d = json.load(f)

        exclude_ids = [e_d[e] for e in short_LaAc if e in e_d]
        other_ids = []
        for k in e_d:
            if e_d[k] not in exclude_ids:
                other_ids.append(e_d[k])

    label_elements = []
    for i in range(n_samples):
        fff = np.random.choice(other_ids,3,replace=False)
        label_elements.append(fff)
    label_elements = np.array(label_elements)

    # ix = np.random.choice(len(candidate_element_comb),n_samples)
    # label_elements = candidate_element_comb[ix]

    return [label_sp,label_elements,z] 

def generate_crystal_cif(generator,lat_dim,n_samples,candidate_element_comb,aux_data,FLAGS):
    gen_inputs = generate_latent_inputs(lat_dim,n_samples,candidate_element_comb,aux_data,FLAGS)
    spacegroup,formulas = gen_inputs[0],gen_inputs[1]
    spacegroup = spacegroup.reshape(-1,)
    
    aux_data_tmp = []
    for inp in aux_data:
        if isinstance(inp, torch.Tensor):
            inp = inp.to(FLAGS.device)
        else:
            inp = inp
        aux_data_tmp.append(inp)
    aux_data = aux_data_tmp #SAME DEVICE
         
    sp_d = aux_data[-2]
    rsp = {sp_d[k]:k for k in sp_d}
    spacegroup = [rsp[ix] for ix in spacegroup]

    with open('cubic-elements-dict.json', 'r') as f:
       e_d = json.load(f)
       re = {e_d[k]:k for k in e_d}
    arr_comb = []
    for i in range(n_samples):
        arr_comb.append([re[e] for e in formulas[i]])

    label_sp,label_elements,z = gen_inputs
    gen_inputs = list(map(lambda inp: torch.from_numpy(inp).to(FLAGS.device).type(torch.float32), [label_sp,label_elements,z])) #return tensors

    generator.eval()
    with torch.inference_mode():
        coords, arr_lengths = generator(gen_inputs)
        
    coords = coords*aux_data[4] + aux_data[4]
    coords = coords.detach().cpu().numpy()
    coords = np.rint(coords/0.25)*0.25
#     print('Coordinates!!!!')
#     print(coords)
    print('Coordinates Done!')
# exit(arr_lengths)
    # arr_angles = arr_angles*aux_data[3]+aux_data[3]
    arr_lengths = arr_lengths*aux_data[2] + aux_data[2]
    arr_lengths = arr_lengths.detach().cpu().numpy()

    if os.path.exists('generated_mat/sample-%d/'%int(FLAGS.sample_size)):
        os.system('rm -rf generated_mat/sample-%d/'%int(FLAGS.sample_size))
#     os.system('mkdir generated_mat/sample-%d/'%int(FLAGS.sample_size))

    if os.path.exists('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size)):
        os.system('rm -rf generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size))
#     os.system('mkdir generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size))

    os.makedirs(os.path.join('generated_mat', 'sample-%d'%(int(FLAGS.sample_size)), 'generated-cifs'), exist_ok=True)
    os.makedirs(os.path.join('generated_mat', 'sample-%d'%(int(FLAGS.sample_size)), 'tmp-symmetrized-cifs'), exist_ok=True)
    os.makedirs(os.path.join('generated_mat', 'sample-%d'%(int(FLAGS.sample_size)), 'tmp-charge-cifs'), exist_ok=True)
        
    for i in tqdm.tqdm(range(n_samples)):
#     def make_one_cif(i):
        f = open('/Scr/hyunpark/ArgonneGNN/cubicgan_modified/data/cif-template.txt', 'r')
        template = f.read()
        f.close()

        lengths = arr_lengths[i][0]
        lengths = [lengths]*3

        angles = [90.0,90.0,90.0]

        template = template.replace('SYMMETRY-SG', spacegroup[i])
        template = template.replace('LAL', str(lengths[0]))
        template = template.replace('LBL', str(lengths[1]))
        template = template.replace('LCL', str(lengths[2]))
        template = template.replace('DEGREE1', str(angles[0]))
        template = template.replace('DEGREE2', str(angles[1]))
        template = template.replace('DEGREE3', str(angles[2]))
        f = open('/Scr/hyunpark/ArgonneGNN/cubicgan_modified/data/symmetry-equiv/%s.txt'%spacegroup[i].replace('/','#'), 'r')
        sym_ops = f.read()
        f.close()

        template = template.replace('TRANSFORMATION\n', sym_ops)

        for j in range(3):
            row = ['',arr_comb[i][j],arr_comb[i][j]+str(j),\
                str(coords[i][j][0]),str(coords[i][j][1]),str(coords[i][j][2]),'1']
            row = '  '.join(row)+'\n'
            template+=row

#         os.makedirs(os.path.join('generated_mat', 'sample-%d'%(int(FLAGS.sample_size)), 'generated-cifs'), exist_ok=True)
#         os.makedirs(os.path.join('generated_mat', 'sample-%d'%(int(FLAGS.sample_size)), 'tmp-symmetrized-cifs'), exist_ok=True)
#         os.makedirs(os.path.join('generated_mat', 'sample-%d'%(int(FLAGS.sample_size)), 'tmp-charge-cifs'), exist_ok=True)

        template += '\n'
        f = open('generated_mat/sample-%d/generated-cifs/%s---%d.cif'%(int(FLAGS.sample_size),spacegroup[i].replace('/','#'),i),'w')
        f.write(template)
        f.close()
        
#     p_imap(make_one_cif, np.arange(n_samples))

def charge_check(crystal):
    # oxidation_states
    elements = list(crystal.composition.as_dict().keys())

    oxi = {}
    for e in elements:
        oxi[e] = Element(e).oxidation_states
    res = []
    if len(oxi) == 3:
        for i in range(len(oxi[elements[0]])):
            for j in range(len(oxi[elements[1]])):
                for k in range(len(oxi[elements[2]])):
                    d = {elements[0]:oxi[elements[0]][i], elements[1]:oxi[elements[1]][j], elements[2]:oxi[elements[2]][k]}
                    crystal.add_oxidation_state_by_element(d)
                    if crystal.charge==0.0:
                        crystal.remove_oxidation_states()
                        res.append(d)
                    crystal.remove_oxidation_states()

    return res

def process(cif, FLAGS):
    sp = cif.split('---')[0].replace('#','/')
    i = int(cif.split('---')[1].replace('.cif',''))
    try:
        crystal = Structure.from_file('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size)+cif)
        
        formula = crystal.composition.reduced_formula
        sg_info = crystal.get_space_group_info(symprec=0.1)

        if sp == sg_info[0]:
            #only valid cif
            crystal.to(fmt='cif',\
             filename='generated_mat/sample-%d/tmp-symmetrized-cifs/%d-%s-%d-%d.cif'%\
             (int(FLAGS.sample_size),len(crystal),formula,sg_info[1],i),symprec=0.1)
            #charge
            res = charge_check(crystal)
            if len(res) > 0:
                crystal.to(fmt='cif',\
                  filename='generated_mat/sample-%d/tmp-charge-cifs/%d-%s-%d-%d.cif'%\
                  (int(FLAGS.sample_size),len(crystal),formula,sg_info[1],i),symprec=0.1)


    except Exception as e:
#         print(e)
        pass
     
def run():
    
    seed_everything(1)
    
    parser = argparse.ArgumentParser()
    data_path = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/mofs_processed.pickle"
    model_path = "/Scr/hyunpark/ArgonneGNN/argonne_gnn/crystals"
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--train_frac', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--data_dir_crystal', type=str, default=data_path)
    parser.add_argument('--model_dir_crystal', type=str, default=model_path)
    parser.add_argument('--model_filename', type=str, default="gan.pt")
#     parser.add_argument('--batch_size', type=int, default=len(os.listdir(data_path)))
    parser.add_argument('--sample_size', type=int, default=64)
    parser.add_argument('--data_norm', type=bool, default=False)
    parser.add_argument('--norm', action="store_true", help="group norm")
    parser.add_argument('--lat_dim', type=int, default=128)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--d_repeat', type=int, default=5)
    parser.add_argument('--pgcgm', type=bool, default=False)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--spectral_norm', action="store_true")
    parser.add_argument('--use_artifacts', action="store_true")
    parser.add_argument('--dataset', type=str, default="gandata")
    parser.add_argument('--make_data', action="store_true", help="force making data")
    parser.add_argument('--mode', type=str, default="train", choices=["train","sample"])
    args = parser.parse_args()
    FLAGS = args
   
    ###DDP###
    is_distributed = init_distributed()
    local_rank = get_local_rank()

    if not dist.is_initialized() or dist.get_rank() == 0:
#             save_dir.mkdir(parents=True, exist_ok=True)
        wandb_run = wandb.init(name=os.path.splitext(FLAGS.model_filename)[0], entity="argonne_gnn", project='internship',
                                     settings=wandb.Settings(start_method="fork"),
                                     id=None,
                                     dir=None,
                                     resume='allow',
                                     anonymous='must')
    
    datamodule = DataModuleCrystal(opt=FLAGS) #   
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    print("0")
    dataset = iter(train_loader).next() #all dataset!
#     print(dataset[:8].arr_coords.shape) #Not possible: i.e. returns [Data(aux_data=[7], arr_element=[3], arr_coords=[3, 3], arr_lengths=[1], arr_angles=[3]), Data(aux_data=[7], arr_element=[3], arr_coords=[3, 3], arr_lengths=[1], arr_angles=[3]), ...]
#     print(dataset[0].arr_element, dataset[0].arr_coords, dataset[0].arr_lengths, dataset[0].arr_angles, ) # arr_sp, arr_element, arr_coords, arr_lengths, arr_angles
    print("1")

#     #load dataset and build models
#     DATA = dataset[:] #List[torchgeom.Data]
#     AUX_DATA = dataset[0].aux_data #extract AUX_DATA (same for all datasets)
#     n_element = AUX_DATA[0]
#     n_spacegroup = AUX_DATA[1]
#     spacegroups = AUX_DATA[-2]
#     device = torch.cuda.current_device()
    device = torch.device("cpu")
#     sp_info = sp_lookup(device, spacegroups) #PGCGM
#     print(sp_info.symm_op_collection)
    print("2")

#     candidate_element_comb = DATA[1]
    n_discriminator = FLAGS.d_repeat
    
#     atom_embedding = np.load('/Scr/hyunpark/ArgonneGNN/argonne_gnn/cubic-elements-features.npy') #Pre-made!
#     atom_embedding = np.load('/Scr/hyunpark/ArgonneGNN/argonne_gnn/crystals/cubic-elements-features.npy') #Pre-made!
    atom_embedding = np.array(list(raw_features.values())).astype(np.float64)
    print("3")

    atom_embedding = torch.from_numpy(atom_embedding).type(torch.float32).to(device)
    print("4")

#     d_model = Discriminator(FLAGS)
#     g_model = build_generator(n_element=63, n_spacegroup=123, atom_embedding=atom_embedding, lat_dim=FLAGS.lat_dim, FLAGS=FLAGS) #.to(device)
#     d_model = build_discriminator(FLAGS) #.to(device)
    gan_model = GAN(g_linear_block_channels=[256, 128, 64, 32, 16, 3], atom_embedding=atom_embedding, lat_dim=FLAGS.lat_dim,
                                        d_linear_block_channels=[17, 128, 256, 256, 128, 64, 1]).to(device)
    gan_model.to(device)
    print("5")

    ###Dist training###
#     if is_distributed:         
#         nproc_per_node = torch.cuda.device_count()
#         affinity = set_affinity(local_rank, nproc_per_node)
#     increase_l2_fetch_granularity()
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    print("6")
    print(local_rank, world_size)
    
    #DDP Model
    if dist.is_initialized():
        gan_model = DistributedDataParallel(gan_model, device_ids=[local_rank], output_device=local_rank)
        gan_model._set_static_graph()
        gan_model.train()
    print(f"DDP is enabled {dist.is_initialized()} and this is local rank {local_rank}!!")    

#     import inspect
#     print(inspect.getfullargspec(Generator.__init__))
    d_optimizer = torch.optim.AdamW(gan_model.discriminator.parameters(), lr=FLAGS.learning_rate)
#     d_scheduler = torch.optim.AdamW(gan_model.discriminator.parameters(), lr=0.00001, betas=(0.5, 0.9))
    g_optimizer = torch.optim.AdamW(gan_model.generator.parameters(), lr=FLAGS.learning_rate)
#     g_scheduler = torch.optim.AdamW(gan_model.generator.parameters(), lr=0.00001)

#     X_real = generate_real_samples(DATA, FLAGS.sample_size, n_element, n_spacegroup)
    
#     print(X_real)
#     d_hist, g_hist = [],[]
#     bat_per_epo = int(DATA[0].shape[0] / FLAGS.batch_size)
    model_path_and_name = os.path.join(FLAGS.model_dir_crystal, FLAGS.model_filename)
    model_path_and_name = os.path.splitext(model_path_and_name)[0] + ".pt"
#     logger = WandbLogger(name=os.path.splitext(FLAGS.model_filename)[0], entity="argonne_gnn", project='internship')
    start_epoch, best_loss = load_states(gan_model, g_optimizer, d_optimizer, model_path_and_name, device, FLAGS, None) if FLAGS.resume else (0, 1e5)

    if FLAGS.mode in ["train"]:
#         idx_list = np.arange(len(DATA))
        os.environ["WANDB_CACHE_DIR"] = os.getcwd()
#         logger.start_watching(gan_model)
        wandb_run.watch(gan_model)
        
        for epoch in range(start_epoch, FLAGS.num_epochs):
            if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            gan_model.train()
            loss_ = 0.
            
            for batch in train_loader:
                dataset = batch
#                 DATA = dataset[:] #NOT THIS! List[torchgeom.Data]
                DATA = dataset
                ix = np.arange(len(DATA))

                X_real = generate_real_samples(DATA, FLAGS.sample_size, ix)
#                 print(X_real)
                d_loss, g_loss = train_step(X_real, device, model_path_and_name, gan_model, atom_embedding, FLAGS, n_discriminator, d_optimizer, g_optimizer)
                loss_ += g_loss

                if local_rank == 0:
                    print(cf.on_yellow(f"Epoch {epoch} ... Discriminator: {d_loss}, Generator: {g_loss}"))
                    wandb_run.log({'rank0_d_loss': d_loss})
                    wandb_run.log({'rank0_g_loss': g_loss})

            loss_ /= len(train_loader)        
            wandb_run.log({'rank0_epoch_g_loss': loss_})
            
            if loss_ < best_loss:
                gan_model.eval()
                save_states(gan_model, g_optimizer, d_optimizer, epoch, loss_, model_path_and_name)
                print(f"{os.path.splitext(FLAGS.model_filename)[0]}_model_objects")
#                 logger.log_artifacts(name=f"{os.path.splitext(FLAGS.model_filename)[0]}_model_objects", dtype="pytorch_gan_models", path_and_name=model_path_and_name)   
                
                artifact = wandb.Artifact(name=f"{os.path.splitext(FLAGS.model_filename)[0]}_model_objects", type="pytorch_gan_models")
                artifact.add_file(str(model_path_and_name)) #which directory's file to add; when downloading it downloads directory/file
                wandb_run.log_artifact(artifact)
    
                best_loss = loss_
                gan_model.train()
                    
            if (epoch % 50 == 0 and local_rank == 0 and epoch != 0) or (local_rank == 0 and epoch == (FLAGS.num_epochs-1) and epoch != 0):
#                 sample_size = FLAGS.sample_size
#                 g_model = gan_model.generator
#                 FLAGS.sample_size = 500
# #                 os.system(f"rm generated_mat/sample-{int(FLAGS.sample_size)}/generated-cifs/*")
#                 generate_crystal_cif(g_model, FLAGS.lat_dim, FLAGS.sample_size, None, AUX_DATA,FLAGS)
#                 gen_cifs = os.listdir('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size))
# #                 os.system(f"rm generated_mat/sample-{int(FLAGS.sample_size)}/tmp-charge-cifs/*")
# #                 list(map(lambda inp: functools.partial(process, FLAGS=FLAGS)(cif=inp), gen_cifs))
#                 p_map(lambda inp: functools.partial(process, FLAGS=FLAGS)(cif=inp), gen_cifs) #https://github.com/swansonk14/p_tqdm
#                 val_cifs = os.listdir('generated_mat/sample-%d/tmp-charge-cifs/'%int(FLAGS.sample_size))
                
# #                 logger.log_metrics({'valid_percentage': 100*len(val_cifs)/len(gen_cifs)})
#                 wandb_run.log({'valid_percentage': 100*len(val_cifs)/len(gen_cifs)})
#                 print(cf.red(f"{100*len(val_cifs)/len(gen_cifs)} percentent is valid..."))
#                 FLAGS.sample_size = sample_size
#                 print(FLAGS.sample_size)
                print()
                print("Reached here!")
                
        FLAGS.use_artifacts = False
        final_epoch, final_loss = load_states(gan_model, g_optimizer, d_optimizer, model_path_and_name, device, FLAGS, None)
        save_states(gan_model, g_optimizer, d_optimizer, final_epoch, final_loss, model_path_and_name) #for Wandb
#         logger.log_artifacts(name=f"{os.path.splitext(FLAGS.model_filename)[0]}_model_objects", dtype="pytorch_gan_models", path_and_name=model_path_and_name)
#         logger.experiment.finish()
        
    elif FLAGS.mode in ["sample"] and FLAGS.resume:
        load_states(gan_model, g_optimizer, d_optimizer, model_path_and_name, device, FLAGS, logger)
        g_model = gan_model.generator
        g_model.eval()
#         os.system(f"rm generated_mat/sample-{int(FLAGS.sample_size)}/generated-cifs/*")
        generate_crystal_cif(g_model, FLAGS.lat_dim, FLAGS.sample_size, None, AUX_DATA,FLAGS)
        gen_cifs = os.listdir('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size))
#         os.system(f"rm generated_mat/sample-{int(FLAGS.sample_size)}/tmp-charge-cifs/*")
        p_map(lambda inp: functools.partial(process, FLAGS=FLAGS)(cif=inp), gen_cifs)
        val_cifs = os.listdir('generated_mat/sample-%d/tmp-charge-cifs/'%int(FLAGS.sample_size))
        print(cf.red(f"{100*len(val_cifs)/len(gen_cifs)} percentent is valid..."))
        
if __name__ == "__main__":
    run()

    
    
    
    
    
    


