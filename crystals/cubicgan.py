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
from torch_geometric.nn import MessagePassing, radius_graph
from torch_scatter import scatter
from transformers import AdamW
import argparse
import pathlib
roots = pathlib.Path(__file__).parent.parent
sys.path.append(roots)
from crystals.dataloader import *
#from dataloader import *
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

if torch.cuda.is_available():
    import torch.cuda as t
else:
    import torch as t

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        t.manual_seed_all(seed)
    
LaAc = ['Ac', 'Am', 'Bk', 'Ce', 'Cf', 'Cm', 'Dy', 'Er', 'Es', 'Eu', 'Fm', 'Gd', 'Ho', 'La', 'Lr', 'Lu', 'Md', 'Nd', 'No', 'Np', 'Pa', 'Pm', 'Pr', 'Pu', 'Sm', 'Tb', 'Th', 'Tm', 'U', 'Yb']
short_LaAc = ['Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr']
    
def build_discriminator(FLAGS: argparse.ArgumentParser):
    Dense = torch.nn.Linear
    LeakyReLU = torch.nn.LeakyReLU
    Conv1D = torch.nn.Conv1d
    Dropout = torch.nn.Dropout
    Flatten = torch.nn.Flatten
    Norm = torch.nn.GroupNorm
    
    class ConvModule(torch.nn.Module):
        #SpectralNorm: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=fbd5ec28ff95db8eef2de13ed96b839db08f2069&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7933332d6a33542f436f7572736572612d446565702d4c6561726e696e672f666264356563323866663935646238656566326465313365643936623833396462303866323036392f4275696c64253230426173696325323047656e65726174697665253230416476657273617269616c2532304e6574776f726b732532302847414e73292f5765656b253230332532302d253230576173736572737465696e25323047414e73253230776974682532304772616469656e7425323050656e616c74792f534e47414e2e6970796e62&logged_in=false&nwo=y33-j3T%2FCoursera-Deep-Learning&path=Build+Basic+Generative+Adversarial+Networks+%28GANs%29%2FWeek+3+-+Wasserstein+GANs+with+Gradient+Penalty%2FSNGAN.ipynb&platform=android&repository_id=324327175&repository_type=Repository&version=98
        #SpectralNorm on Discriminator only:https://blog.ml.cmu.edu/2022/01/21/why-spectral-normalization-stabilizes-gans-analysis-and-improvements/
        def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
            super().__init__()
            if FLAGS.spectral_norm:
                self.add_module("conv_module", torch.nn.Sequential(spectral_norm(Conv1D(in_channels, out_channels, kernel_size, padding=padding)), 
                                                               Dropout(0.), LeakyReLU(0.2, inplace=True)))
            else:
                if not FLAGS.norm:
                    self.add_module("conv_module", torch.nn.Sequential(Conv1D(in_channels, out_channels, kernel_size, padding=padding), 
                                                               Dropout(0.), LeakyReLU(0.2, inplace=True)))   
                else:
                    self.add_module("conv_module", torch.nn.Sequential(Conv1D(in_channels, out_channels, kernel_size, padding=padding), 
                                                               Norm(1, out_channels), LeakyReLU(0.2, inplace=True)))   
                
        def forward(self, inputs: torch.Tensor):
            return self.conv_module(inputs)
            
    class LinearModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.add_module("linear_module", torch.nn.Sequential(Dense(in_channels, out_channels), 
                                                               LeakyReLU(0.2, inplace=True)))
        
        def forward(self, inputs: torch.Tensor):
            return self.linear_module(inputs)
    
    class Discriminator(torch.nn.Module):
        def __init__(self, conv_block_channels=[28,128,512,1024,1024,2048,4096], 
                     conv_block_kernels=[1,1,1,3,2,2],
                     linear_block_channels=[4096,1024,512,256,32]):
            super().__init__()
            
            conv_list = torch.nn.ModuleList([])
            input_cs = conv_block_channels[:-1]
            output_cs = conv_block_channels[1:]
            for idx, (inc, outc, k) in enumerate(zip(input_cs, output_cs, conv_block_kernels)):
                convo = ConvModule(inc, outc, k) if idx != 3 else ConvModule(inc, outc, k, padding="same")
                conv_list.append(convo)
#             print(conv_list)
            self.add_module("conv_block", torch.nn.Sequential(*conv_list))
#             self.add_module("conv_block", conv_list) #as modulelist
            self.add_module("flat", torch.nn.Flatten())    
                
            linear_list = torch.nn.ModuleList([])
            input_cs = linear_block_channels[:-1]
            output_cs = linear_block_channels[1:]
            for inc, outc in zip(input_cs, output_cs):
                linear_list.append(LinearModule(inc, outc))
            self.add_module("linear_block", torch.nn.Sequential(*linear_list))
            
            self.add_module("final_linear", torch.nn.Linear(output_cs[-1], 1))
            
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        
        def forward(self, inputs: torch.Tensor):
            b, c, l = inputs.size()
            x = inputs
            assert c==28 and l==3, "input format is wrong..."
            x = self.conv_block(x) #BCL -> BC1
#             for idx, module in enumerate(self.conv_block):
# #                 import pdb; pdb.set_trace()
#                 x = module(x)
#                 print(idx)

            x = self.flat(x) #BC1 -> BC
            x = self.linear_block(x) #BC->BD
            x = self.final_linear(x) #BD->B1
            return x
        
    return Discriminator

def build_generator(n_element=60, n_spacegroup=123, atom_embedding: torch.Tensor=None, lat_dim: int=None, FLAGS=None):
    Dense = torch.nn.Linear
    ReLU = torch.nn.ReLU
    Tanh = torch.nn.Tanh
    LeakyReLU = torch.nn.LeakyReLU
    Conv1D = torch.nn.Conv1d
    Conv2D = torch.nn.Conv2d
    Dropout = torch.nn.Dropout
    Flatten = torch.nn.Flatten
    Conv2DTranspose = torch.nn.ConvTranspose2d
    Norm = torch.nn.GroupNorm
    
    class SpaceGroup(torch.nn.Module):
        def __init__(self, ):
            super().__init__()
#             self.add_module("spacegroup", torch.nn.Sequental(torch.nn.Embedding(n_spacegroup, 64), Conv1D(96,1), torch.nn.ReLU(inplace=True), Flatten())
            self.embedding = torch.nn.Embedding(n_spacegroup, 64)
            self.conv = Conv1D(64, 96, 1)
            self.act = ReLU()
            self.flat = Flatten()
        
        def forward(self, inputs: torch.Tensor):
            x = inputs.long()
            x = self.embedding(x) #B,1,64
            x = x.permute(0,2,1) #B,64,1
            x = self.conv(x) #B,96,1
            x = self.act(x)
            x = self.flat(x) #B,96
            return x
            
    class Element(torch.nn.Module):
        def __init__(self, ):
            super().__init__()
#             self.add_module("spacegroup", torch.nn.Sequental(torch.nn.Embedding(n_spacegroup, 64), Conv1D(96,1), torch.nn.ReLU(inplace=True), Flatten())
            self.embedding = torch.nn.Embedding(n_element, 23)
            self.conv = Conv1D(23, 128, 1) #3 atoms
            self.act = ReLU()
            self.flat = Flatten()
        
        def forward(self, inputs: torch.Tensor):
            x = inputs.long()
            b,c = x.size()
            x = inputs.long().view(-1) #B*3
            x = atom_embedding.index_select(index=x, dim=0).view(-1, 3, 23) #B,3,23 
#             x = self.embedding(x) #.view(b,c,-1) #B,3,23 
#             print(x)
            x = x.permute(0,2,1) #B,23,3
            x = self.conv(x) #B,128,3
            x = self.act(x)
            x = self.flat(x) #B,128*3
            return x
    
    class Latent(torch.nn.Module):
        def __init__(self, ):
            super().__init__()
            self.linear = Dense(lat_dim, 256)
            self.act = ReLU()
            self.flat = Flatten()
        
        def forward(self, inputs: torch.Tensor):
            x = self.linear(inputs) #B,256
            x = self.act(x)
            x = self.flat(x) #B,256
            return x
        
    class ConvModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            if FLAGS.spectral_norm:
                self.add_module("conv_module", torch.nn.Sequential(spectral_norm(Conv2D(in_channels, out_channels, kernel_size)), 
                                                               ReLU(inplace=True),
                                                               Dropout(0.)))
            else:
                if not FLAGS.norm:
                    self.add_module("conv_module", torch.nn.Sequential(Conv2D(in_channels, out_channels, kernel_size), 
                                                               ReLU(inplace=True),
                                                               Dropout(0.)))
                else:
                    self.add_module("conv_module", torch.nn.Sequential(Conv2D(in_channels, out_channels, kernel_size), 
                                                               ReLU(inplace=True),
                                                               Norm(1, out_channels)))
        
        def forward(self, inputs: torch.Tensor):
            return self.conv_module(inputs)
        
    class ConvTransModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, padding="valid"):
            super().__init__()
            if FLAGS.spectral_norm:
                self.add_module("conv_module", torch.nn.Sequential(spectral_norm(Conv2DTranspose(in_channels, out_channels, kernel_size)), 
                                                               ReLU(inplace=True),
                                                               Dropout(0.)))
            else:
                if not FLAGS.norm:
                    self.add_module("conv_module", torch.nn.Sequential(Conv2DTranspose(in_channels, out_channels, kernel_size), 
                                                               ReLU(inplace=True),
                                                               Dropout(0.)))
                else:
                    self.add_module("conv_module", torch.nn.Sequential(Conv2DTranspose(in_channels, out_channels, kernel_size), 
                                                               ReLU(inplace=True),
                                                               Norm(1, out_channels)))
        
        def forward(self, inputs: torch.Tensor):
            return self.conv_module(inputs)   
        
    class LinearModule(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.add_module("linear_module", torch.nn.Sequential(Dense(in_channels, out_channels), 
                                                               ReLU(inplace=True)))
        
        def forward(self, inputs: torch.Tensor):
            return self.linear_module(inputs)
        
    class Generator(torch.nn.Module):
        def __init__(self, conv_block_channels=[128, 1024, 1024, 512, 512, 128], 
                     conv_block_kernels=[(1,2),(1,2),(1,1),(1,1),(1,1)],
                     linear_block_channels=[9,30,18,6]):
            super().__init__()
            self.spacegroup = SpaceGroup()
            self.element = Element()
            self.latent = Latent()

            self.pre_linear = Dense(736, 384)
            
            conv_list = torch.nn.ModuleList([])
            input_cs = conv_block_channels[:-1]
            output_cs = conv_block_channels[1:]
            for idx, (inc, outc, k) in enumerate(zip(input_cs, output_cs, conv_block_kernels)):
                convo = ConvTransModule(inc, outc, k) if idx < 2 else ConvModule(inc, outc, k)
                conv_list.append(convo)
            self.add_module("conv_block", torch.nn.Sequential(*conv_list))
            self.add_module("post_conv", torch.nn.Sequential(Conv2D(128, 1, (1,1)), 
                                                               Tanh()))

            self.flat = Flatten()
            
            linear_list = torch.nn.ModuleList([])
            input_cs = linear_block_channels[:-1]
            output_cs = linear_block_channels[1:]
            for inc, outc in zip(input_cs, output_cs):
                linear_list.append(LinearModule(inc, outc))
            self.add_module("linear_block", torch.nn.Sequential(*linear_list))

            self.post_linear = Dense(6,1)      
#             self.act0 = Tanh() #Already done in post_conv
            self.act1 = Tanh()
                            
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                            
        def forward(self, inputs: List[torch.Tensor]):
            sp, el, lat = inputs
            sp, el, lat = list(map(lambda func, inp: func(inp), (self.spacegroup, self.element, self.latent), (sp, el, lat) ))
#             print(sp, el, lat)
#             print(list(map(lambda tensor: torch.isnan(tensor).any(), [sp, el, lat] )))
            b = len(sp)
            x = torch.cat([sp,el,lat], dim=-1) #B,736 (256+128*3+96)
            x = self.pre_linear(x) #B,384
            x = torch.nn.functional.relu(x) #B,dim
            x = x.view(b,3,1,-1).permute(0,3,1,2) #B,3,1,128 -> (B,128,3,1)
            
            x = self.conv_block(x)
            x = self.post_conv(x)
#             x = self.act0(x)
                            
            coords = x.view(b,3,3)
            lengths = self.flat(coords) #B,dim
            lengths = self.linear_block(lengths) #B, 6     
            lengths = self.post_linear(lengths) #B, 1
            lengths = self.act1(lengths)
            return coords, lengths #(B,3,3) (B,1)
                            
    return Generator

class GAN(torch.nn.Module):
    def __init__(self, g_model: torch.nn.Module, d_model: torch.nn.Module):
        super().__init__()
        self.generator = g_model()
        self.discriminator = d_model()
        
    def forward(self, x):
        return x
    
def calc_grad_penalty_cubic(netD, device, real_samples, fake_samples, FLAGS):
    """ Calculates the gradient penalty.
    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    https://github.com/EmilienDupont/wgan-gp/blob/master/training.py
    """
    #-------------
    #credits to https://keras.io/examples/generative/wgan_gp/
    #-------------
    # get the interplated image
    alpha = torch.randn(FLAGS.sample_size, 1, 1).to(real_samples)
    diff = fake_samples - real_samples
    interpolated = real_samples.data + alpha * diff.data #just a leaf-tensor
    
    interpolated.requires_grad_()
    netD.train()
    [p.requires_grad_(False) for p in netD.parameters()]
    pred = netD(interpolated)
    [p.requires_grad_(True) for p in netD.parameters()]

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = torch.autograd.grad(pred, interpolated, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
    grads = grads.view(FLAGS.sample_size, -1)
    grad_norm_sqrt = grads.norm(dim=-1)
    gp = (grad_norm_sqrt - 1.0).pow(2).mean()
    return gp
    
def calc_grad_penalty_pgcgm(netD, device, real_data, fake_data):
    real_mat, real_symm = real_data
    fake_mat, fake_symm = fake_data
    batch_size = real_mat.shape[0]

    alpha = torch.normal(0.0, 1.0, size=(batch_size,1,1,1)).to(device)
    interpolated_mat = alpha * real_mat + ((1 - alpha) * fake_mat)
    interpolated_mat = interpolated_mat.to(device)
    interpolated_mat = autograd.Variable(interpolated_mat, requires_grad=True)

    alpha = torch.normal(0.0, 1.0, size=(batch_size,1,1,1)).to(device)
    interpolated_symm = alpha * real_symm + ((1 - alpha) * fake_symm)
    interpolated_symm = interpolated_symm.to(device)
    interpolated_symm = autograd.Variable(interpolated_symm, requires_grad=True)

    pred_interpolates = netD(interpolated_mat, interpolated_symm)

    gradients = autograd.grad(
        outputs=pred_interpolates,
        inputs=[interpolated_mat, interpolated_symm],
        grad_outputs=torch.ones(pred_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    
    gradients0 = gradients[0].contiguous().view(batch_size, -1)
    gradients1 = gradients[1].contiguous().view(batch_size, -1)
    gradient_penalty = ((gradients0.norm(2, dim=1) - 1) ** 2).mean() + ((gradients1.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def generate_real_samples(dataset, n_samples, n_element,n_spacegroup, ix):
#     symmetry,elements,coords,lengths,angles #arr_sp, arr_element, arr_coords, arr_lengths, arr_angles
#     X_symmetry,X_elements,X_coords,X_lengths,X_angles = symmetry[ix],elements[ix],coords[ix],lengths[ix],angles[ix]
#     return list(map(lambda inp: torch.from_numpy(inp).type(torch.float32), [X_symmetry,X_elements,X_coords,X_lengths,X_angles]))
    X_symmetry = torch.stack([dataset[i].arr_sp for i in ix], dim=0).long() #(Batch,*)
    X_elements = torch.stack([dataset[i].arr_element for i in ix], dim=0).long() #(Batch,*)
    X_coords = torch.stack([dataset[i].arr_coords for i in ix], dim=0).float() #(Batch,*)
    X_lengths = torch.stack([dataset[i].arr_lengths for i in ix], dim=0).float() #(Batch,*)
    X_angles = torch.stack([dataset[i].arr_angles for i in ix], dim=0).float() #(Batch,*)
    return X_symmetry,X_elements,X_coords,X_lengths,X_angles
    
def generate_fake_lables(n_samples, aux_data):
    label_sp = np.random.choice(aux_data[1], n_samples, p=aux_data[-1]).reshape(-1,1)
    label_elements = [] #nsamples,
    for i in range(n_samples):
        fff = np.random.choice(aux_data[0],3,replace=False)
        label_elements.append(fff)
    label_elements = np.array(label_elements) #(nsamples, 3)
    return list(map(lambda inp: torch.from_numpy(inp).type(torch.float32), [label_sp,label_elements])) #(B,*)

# Define the loss functions to be used for discriminator
# This should be (fake_loss - real_loss)
def discriminator_loss(real_spl, fake_spl):
    real_loss = torch.mean(real_spl)
    fake_loss = torch.mean(fake_spl)
    return fake_loss - real_loss

# Define the loss functions to be used for generator
def generator_loss(fake_spl):
    return -torch.mean(fake_spl)

def train_step(X_real: List[torch.Tensor], device, model_path_and_name, gan_model, atom_embedding, FLAGS, n_discriminator, d_optimizer, g_optimizer, AUX_DATA):
    d_model = gan_model.discriminator
    g_model = gan_model.generator
    
    #symmetry,elements,coords,lengths,angles
    real_crystal = [tensor.type(torch.float32).to(device) for tensor in X_real] #What about device ?
    sp_onehot = torch.nn.functional.one_hot(real_crystal[0].long(), num_classes=3).view(-1, 3, 1) #(n_samples=-1, 3, 1)
    e=real_crystal[1].long().view(-1) #(n_samples, 3) -> (nsamples*3)
    e=atom_embedding.index_select(index=e, dim=0).view(-1, 3, 23) #(nsamples, 3, dim)
    coords = real_crystal[2]
#     print(real_crystal)
    l = real_crystal[3].view(-1,1,1).expand(-1,3,-1) #(b,3,1)
    real_crystal = torch.cat((coords,e,l,sp_onehot), dim=-1).permute(0,2,1).to(device) #DONE: b,3,Dim=28 -> b,28,3
#     print(coords,e,l,sp_onehot)

    original_sample_size = FLAGS.sample_size
    FLAGS.sample_size = real_crystal.size(0)
    
    for _ in range(n_discriminator):
        d_model.train()
        g_model.eval()
        noise = torch.randn(FLAGS.sample_size, FLAGS.lat_dim).to(device) #Device?
        fake_labels = generate_fake_lables(FLAGS.sample_size, AUX_DATA)
#         print(fake_labels)
        sp_inputs, ele_inputs = fake_labels
#         print(sp_inputs, ele_inputs, noise)
        fake_crystal = g_model([sp_inputs.to(device), ele_inputs.to(device), noise])
#         print(fake_crystal)

        sp_onehot = torch.nn.functional.one_hot(fake_labels[0].long(),
                        num_classes=3).view(-1, 3, 1).to(device) #(batch_size=-1, 3, 1)
        e=fake_labels[1].long().view(-1).to(device)
        e=atom_embedding.index_select(index=e, dim=0).view(-1, 3, 23) #(nsamples, 3, dim)
        coords = fake_crystal[0].to(device) #nsamples,res,3
        l = fake_crystal[1].view(-1,1,1).expand(-1,3,-1).to(device) #(b,3,1); length
        fake_crystal = torch.cat((coords,e,l,sp_onehot), dim=-1).permute(0,2,1) #b,3,Dim=28?
        
        fake_logits, real_logits = d_model(fake_crystal), d_model(real_crystal)
        #get the discriminator loss
        d_loss = discriminator_loss(real_logits, fake_logits)

        gp = calc_grad_penalty_cubic(d_model, device, real_crystal, fake_crystal, FLAGS) #compare concats
#         print(d_loss, gp)
        d_loss = d_loss + 10.0*gp
        
        d_optimizer.zero_grad()
        d_loss.backward()
#         print(d_loss)
        d_optimizer.step()
        
    d_model.eval()
    g_model.train()
    noise = torch.randn(FLAGS.sample_size, FLAGS.lat_dim).to(device)
    fake_labels = generate_fake_lables(FLAGS.sample_size, AUX_DATA)
    sp_inputs, ele_inputs = fake_labels
    fake_crystal = g_model([sp_inputs.to(device), ele_inputs.to(device), noise])
    sp_onehot = torch.nn.functional.one_hot(fake_labels[0].long(),
                                                num_classes=3).view(-1, 3, 1).to(device) #(batch_size=-1, 3, 1)
    e=fake_labels[1].long().view(-1).to(device)
    e=atom_embedding.index_select(index=e, dim=0).view(-1, 3, 23)
    coords = fake_crystal[0].to(device)
    l = fake_crystal[1].view(-1,1,1).expand(-1,3,-1).to(device) #(b,3,1)
    fake_crystal = torch.cat((coords,e,l,sp_onehot), dim=-1).permute(0,2,1) #b,3,Dim=28?

    gen_crystal_logits = d_model(fake_crystal)
    #get the generator loss
    g_loss = generator_loss(gen_crystal_logits)

    g_optimizer.zero_grad()
    g_loss.backward()
#     print(g_loss)
    g_optimizer.step()
#     print(coords,e,l,sp_onehot)
#     model_path_and_name = os.path.splitext(model_path_and_name)[0] + ".pt"
    
#     gan_model.discriminator = d_model
#     gan_model.generator = g_model
    
#     print("Saving a GAN model...")
#     torch.save(gan_model.state_dict(), f"{model_path_and_name}")
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
    if torch.cuda.is_available():
        pngname1 = 'logs/d_loss-%d.png'%(FLAGS.device)
        pngname2 = 'logs/g_loss-%d.png'%(FLAGS.device)
    else:
        pngname1 = 'logs/d_loss-0.png'
        pngname2 = 'logs/g_loss-0.png'
    plt.plot(d_hist)
    plt.xlabel('step (s)')
    plt.savefig(pngname1)
    plt.close()

    plt.plot(g_hist)
    plt.xlabel('step (s)')
    plt.savefig(pngname2)

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
            if torch.cuda.is_available():
                inp = inp.to(FLAGS.device)
            else:
                device = t.device("cpu")
                inp = inp.to(device)
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
    if torch.cuda.is_available():
        gen_inputs = list(map(lambda inp: torch.from_numpy(inp).to(FLAGS.device).type(torch.float32), [label_sp,label_elements,z])) #return tensors
    else:
        gen_inputs = list(map(lambda inp: torch.from_numpy(inp).to(t.device("cpu")).type(torch.float32), [label_sp,label_elements,z])) #return tensors
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
        f = open('./cif-template.txt', 'r')
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
        f = open('../cubicgan_modified/data/symmetry-equiv/%s.txt'%spacegroup[i].replace('/','#'), 'r')
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
    data_path = "../cubicgan_modified/data/trn-cifs"
    model_path = "./crystals"
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_frac', type=float, default=0.95)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--data_dir_crystal', type=str, default=data_path)
    parser.add_argument('--model_dir_crystal', type=str, default=model_path)
    parser.add_argument('--model_filename', type=str, default="gan.pt")
    parser.add_argument('--batch_size', type=int, default=int(len(os.listdir(data_path)) / 10))
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
        wandb_run = wandb.init(name=os.path.splitext(FLAGS.model_filename)[0], 
                                     entity="argonne_gnn", project='internship',
                                     settings=wandb.Settings(start_method="fork"),
                                     id=None,
                                     dir=None,
                                     resume='allow',
                                     anonymous='must')
    
    datamodule = DataModuleCrystal(opt=FLAGS) #    if len(bases)==3 and len(set(bases))==3 and all(occu == 1.0) in dataloader is important for GAN!
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    print("0")
    dataset = iter(train_loader).next() #all dataset!
#     print(dataset[:8].arr_coords.shape) #Not possible: i.e. returns [Data(aux_data=[7], arr_element=[3], arr_coords=[3, 3], arr_lengths=[1], arr_angles=[3]), Data(aux_data=[7], arr_element=[3], arr_coords=[3, 3], arr_lengths=[1], arr_angles=[3]), ...]
#     print(dataset[0].arr_element, dataset[0].arr_coords, dataset[0].arr_lengths, dataset[0].arr_angles, ) # arr_sp, arr_element, arr_coords, arr_lengths, arr_angles
    print("1")

#     #load dataset and build models
    DATA = dataset[:] #List[torchgeom.Data]
    AUX_DATA = dataset[0].aux_data #extract AUX_DATA (same for all datasets)
    n_element = AUX_DATA[0]
    n_spacegroup = AUX_DATA[1]
    spacegroups = AUX_DATA[-2]
    #device = t.current_device()
    if torch.cuda.is_available():
        device = t.current_device()
    else:
        device = t.device("cpu")

#     device = torch.device(FLAGS.device) #WIP: FIX
    sp_info = sp_lookup(device, spacegroups) #PGCGM
#     print(sp_info.symm_op_collection)
    print("2")

#     candidate_element_comb = DATA[1]
    n_discriminator = FLAGS.d_repeat
    
#     atom_embedding = np.load('/Scr/hyunpark/ArgonneGNN/argonne_gnn/cubic-elements-features.npy') #Pre-made!
    #atom_embedding = np.load('/Scr/hyunpark/ArgonneGNN/argonne_gnn/crystals/cubic-elements-features.npy') #Pre-made!
    atom_embedding = np.load('./cubic-elements-features.npy') #Pre-made!
    print("3")

    atom_embedding = torch.from_numpy(atom_embedding).type(torch.float32).to(device)
    print("4")

#     d_model = Discriminator(FLAGS)
    g_model = build_generator(n_element=63, n_spacegroup=123, atom_embedding=atom_embedding, lat_dim=FLAGS.lat_dim, FLAGS=FLAGS) #.to(device)
    d_model = build_discriminator(FLAGS) #.to(device)
    gan_model = GAN(g_model, d_model).to(device)
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
                DATA = dataset[:] #List[torchgeom.Data]
                ix = np.arange(len(DATA))

                X_real = generate_real_samples(DATA, FLAGS.sample_size, n_element, n_spacegroup, ix)
                d_loss, g_loss = train_step(X_real, device, model_path_and_name, gan_model, atom_embedding, FLAGS, n_discriminator, d_optimizer, g_optimizer, AUX_DATA)
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
                sample_size = FLAGS.sample_size
                g_model = gan_model.generator
                FLAGS.sample_size = 500
#                 os.system(f"rm generated_mat/sample-{int(FLAGS.sample_size)}/generated-cifs/*")
                generate_crystal_cif(g_model, FLAGS.lat_dim, FLAGS.sample_size, None, AUX_DATA,FLAGS)
                gen_cifs = os.listdir('generated_mat/sample-%d/generated-cifs/'%int(FLAGS.sample_size))
#                 os.system(f"rm generated_mat/sample-{int(FLAGS.sample_size)}/tmp-charge-cifs/*")
#                 list(map(lambda inp: functools.partial(process, FLAGS=FLAGS)(cif=inp), gen_cifs))
                p_map(lambda inp: functools.partial(process, FLAGS=FLAGS)(cif=inp), gen_cifs) #https://github.com/swansonk14/p_tqdm
                val_cifs = os.listdir('generated_mat/sample-%d/tmp-charge-cifs/'%int(FLAGS.sample_size))
                
#                 logger.log_metrics({'valid_percentage': 100*len(val_cifs)/len(gen_cifs)})
                wandb_run.log({'valid_percentage': 100*len(val_cifs)/len(gen_cifs)})
                print(cf.red(f"{100*len(val_cifs)/len(gen_cifs)} percentent is valid..."))
                FLAGS.sample_size = sample_size
                print(FLAGS.sample_size)
                
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
        
#         print('>%d/%d, d=%.4f g=%.4f' %
#                 (i+1, FLAGS.num_epochs*bat_per_epo, d_loss, g_loss))

#         if i%50==0:
#             d_hist.append(d_loss)
#             g_hist.append(g_loss)

#     plot_history(d_hist, g_hist)
#     g_model.save('models/clean-wgan-generator-%d.h5'%(FLAGS.device))
#     d_model.save('models/clean-wgan-discriminator-%d.h5'%(FLAGS.device))

#     python -m crystals.cubicgan --num_epochs 500 --lat_dim 64 --sample_size 16 --mode train 
#     python -m torch.distributed.run --nnodes=1 --nproc_per_node=gpu --max_restarts=0 --module crystals.cubicgan --num_epochs 500 --lat_dim 64 --sample_size 16 --mode train
    
    
    
    
    
    


# class Discriminator(nn.Module):
#     def __init__(self, args: argparse.ArgumentParser):
#         super(Discriminator, self).__init__()
#         self.pgcgm = args.pgcgm
        
#         self.crystal_block = nn.Sequential(
#                 nn.Conv2d(3,16,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(16,32,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(32,64,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(64,96,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(96,128,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(128,192,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(192,256,2,stride=1),
#                 nn.LeakyReLU(0.2),

#                 nn.Flatten()
#             )

#         self.sp_block = nn.Sequential(
#                 nn.Conv2d(192, 64, 2, 1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(64, 128, 2, 1),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(128, 256, 2, 1),
#                 nn.LeakyReLU(0.2),

#                 nn.Flatten()
#             )

#         if self.pgcgm:
#             self.dense_block = nn.Sequential(
#                     nn.Linear(512, 256),
#                     nn.LeakyReLU(0.2),

#                     nn.Linear(256, 1)
#                 )

#         else:
#             self.dense_block = nn.Sequential(
#                     nn.Linear(256, 256),
#                     nn.LeakyReLU(0.2),

#                     nn.Linear(256, 1)
#                 )

#     def forward_pgcgm(self, crystal, symm_mat):
#         x1 = self.crystal_block(crystal)
#         x2 = self.sp_block(symm_mat)
#         x = torch.cat((x1, x2), 1)
#         x = self.dense_block(x)
#         return x
    
#     def forward_cubic(self, crystal, symm_mat):
#         x1 = self.crystal_block(crystal)
#         x = x1
#         x = self.dense_block(x)
#         return x
    
#     def forward(self, crystal, symm_mat):
#         if self.pgcgm:
#             return self.forward_pgcgm(crystal, symm_mat)
#         else:
#             return self.forward_cubic(crystal, symm_mat)
        
# class Generator(nn.Module):
#     def __init__(self, ele_vec_dim=23, noise_dim=128):
#         super().__init__()
#         self.sp_block = nn.Sequential(
#                 nn.Conv2d(192, 64, 2, 1),
#                 nn.BatchNorm2d(64),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(64, 128, 2, 1),
#                 nn.BatchNorm2d(128),
#                 nn.LeakyReLU(0.2),

#                 nn.Conv2d(128, 256, 2, 1),
#                 nn.BatchNorm2d(256),
#                 nn.LeakyReLU(0.2),

#                 nn.Flatten()
#             )

#         self.ele_block = nn.Sequential(
#                 nn.Conv1d(ele_vec_dim, 64, 2),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(),

#                 nn.Conv1d(64, 128, 2),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU(),

#                 nn.Flatten(),

#                 nn.Linear(128, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU(),

#             )

#         self.noise_block = nn.Sequential(
#                 nn.Linear(noise_dim, 256),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU(),
#             )

#         self.coords_block1 = nn.Sequential(
#                 nn.ConvTranspose2d(512,1024,(2,2),(1,1)),
#                 nn.BatchNorm2d(1024),
#                 nn.ReLU(),

#                 nn.ConvTranspose2d(1024,512,(2,2),(1,1)),
#                 nn.BatchNorm2d(512),
#                 nn.ReLU(),

#                 nn.ConvTranspose2d(512,256,(1,1),(1,1)),
#                 nn.BatchNorm2d(256),
#                 nn.ReLU(),

#                 nn.ConvTranspose2d(256,128,(1,1),(1,1)),
#                 nn.BatchNorm2d(128),
#                 nn.ReLU(),

#                 nn.ConvTranspose2d(128,64,(1,1),(1,1)),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU(),

#                 nn.ConvTranspose2d(64,3,(1,1),(1,1)),
#                 nn.Tanh()
#             )

#         self.length_block = nn.Sequential(
#                 nn.Linear(512,128),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU(),

#                 nn.Linear(128,64),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(),

#                 nn.Linear(64,32),
#                 nn.BatchNorm1d(32),
#                 nn.ReLU(),

#                 nn.Linear(32,16),
#                 nn.BatchNorm1d(16),
#                 nn.ReLU(),

#                 nn.Linear(16,3),
#                 nn.Tanh()
#             )

#     def forward(self, sp_inputs, ele_inputs, z):
#         sp_embedding = self.sp_block(sp_inputs)
#         ele_embedding = self.ele_block(ele_inputs)
#         z_embedding = self.noise_block(z)
        
#         x1 = torch.cat((z_embedding, ele_embedding), 1)
#         x2 = torch.cat((z_embedding, sp_embedding), 1)
        
#         coords = self.coords_block1(x1.view(-1,512,1,1))
#         length = self.length_block(x2)
        
#         return coords,length

