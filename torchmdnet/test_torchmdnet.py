import torch

#Get argparser
def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default=''.join(random.choice(string.ascii_lowercase) for i in range(10)))
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--gpus', action='store_true')
	parser.add_argument('--log', action='store_true')
	parser.add_argument('--plot', action='store_true')

	# data
	parser.add_argument('--train_test_ratio', type=float, default=0.02)
	parser.add_argument('--train_val_ratio', type=float, default=0.03)
	parser.add_argument('--train_frac', type=float, default=1.0)
	parser.add_argument('--warm_up_split', type=int, default=5)
	parser.add_argument('--batches', type=int, default=32)
	parser.add_argument('--test_samples', type=int, default=5) # -1 for all
	parser.add_argument('--test_steps', type=int, default=100)
	parser.add_argument('--data_norm', action='store_true')
	parser.add_argument('--data_dir', type=str, default="/Scr/hyunpark/ArgonneGNN/argonne_gnn/data")
	parser.add_argument('--task', type=str, default="homo")

	# train
	parser.add_argument('--epoches', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--dropout', type=float, default=0)
	parser.add_argument('--resume', action='store_true')
	parser.add_argument('--distributed',  action="store_true")
	parser.add_argument('--low_memory',  action="store_true")
	parser.add_argument('--amp', action="store_true", help="floating 16 when turned on.")
	parser.add_argument('--loss_schedule', '-ls', type=str, choices=["manual", "lrannealing", "softadapt", "relobralo", "gradnorm"], help="how to adjust loss weights.")

	# model
	parser.add_argument('--backbone', type=str, default='torchmdnet', choices=["schnet","physnet","torchmdnet","alignn","dimenet"])

	opt = parser.parse_args()

	return opt
import string; import random; import argparse
opt = get_parser()

#Initialize model
from torchmdnet import torchmdnet
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
m = torchmdnet.create_model(config)

#Get Dataloader
import train.dataloader
dl = train.dataloader.DataModule(opt=opt)
td = dl.train_dataloader()
li=next(iter(td)) #dict
targ = li.pop("targets")

#Run model
result = m(**li)
print(result)
