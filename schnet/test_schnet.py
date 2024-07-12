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
	parser.add_argument('--data_dir', type=str, default="/lus/grand/projects/AIHacks/gpuHack/data/")
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
from schnet import schnet
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
m = schnet.SchNet(**config)

#Get Dataloader
import train.dataloader
dl = train.dataloader.DataModule(opt=opt)
td = dl.train_dataloader()
li=next(iter(td)) #dict
targ = li.pop("targets")

#Run model
result = m(**li)
print(result)
