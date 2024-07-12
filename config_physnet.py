from physnet import physnet
import torch

__all__ = ['BACKBONES', 'BACKBONE_KWARGS']

BACKBONES = {
    'physnet': physnet.Physnet,
}

BACKBONE_KWARGS = {
    'physnet': {
        'dfilter': 128,
        'filter': 128,
        'cutoff': 10,
        'num_residuals': 3,
        'num_residuals_atomic': 2,
        'num_interactions': 5,
        'num_outer_residuals': 1,
        'activation_fn': torch.nn.ReLU(),
        'dmodel': 64,
        'token_embedding_necessary': True,
        'max_num_neighbors': 32,
        'readout': "sum"
    }
}
