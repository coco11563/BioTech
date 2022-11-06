import argparse

import torch

from model.dataset import construct_ds
from model.encoder import construct_gnn
from model.trainer import Wrapper

parser = argparse.ArgumentParser()

parser.add_argument('--encoder', default='gcn',choices=['gcn', 'gat', 'gsage', 'gin'])
parser.add_argument('--encoder_dim', default=128, type=int)
parser.add_argument('--num_features', default=128, type=int)
parser.add_argument('--names_1', default='ADT',choices=['ADT', 'SCT'])
parser.add_argument('--names_2', default='SCT', choices=['ADT', 'SCT'])
parser.add_argument('--l1_w', default=1.0, type=float)
parser.add_argument('--l2_w', default=1.0, type=float)
parser.add_argument('--l3_w', default=1.0, type=float)
parser.add_argument('--l4_w', default=1.0, type=float)
parser.add_argument('--epoches', default=60, type=int)
parser.add_argument('--report_epoch', default=1, type=int)

args = parser.parse_args()


ds1 = construct_ds(args.names_1)
ds2 = construct_ds(args.names_2)
encoder = construct_gnn(args.encoder, ds1.data.x.shape[1] + ds2.data.x.shape[1], (ds1.data.x.shape[1] + ds2.data.x.shape[1]))
decoder = None
wrapper = Wrapper(encoder, decoder, ds1, ds2,args)
wrapper.to(torch.device('cuda:1'))
wrapper.fit()
