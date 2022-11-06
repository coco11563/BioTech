import os.path
import os.path as osp
import pickle

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.data import (InMemoryDataset)
from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx


def load_processed_file(root, name):
    network_file = os.path.join(root, f'{name}.networkx.g')
    # network_file = os.path.join(root, f'{name}.adj.coo.npy')
    feature_file = os.path.join(root, f'{name}.feat.pt')
    label_file = os.path.join(root, 'label.pt')
    # with open(network_file, 'rb') as f:
    #     g = pickle.load(f)
    return {
        # 'network': g,
            'network' : np.load(network_file, allow_pickle=True),
            'attrb' : torch.load(feature_file).numpy(),
            'group' : torch.load(label_file).numpy()
            }


class DomainData(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        # self.root = root
        super(DomainData, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        '''
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)
        '''
        # net = sio.loadmat(osp.join(self.raw_dir, self.name))
        net = load_processed_file(self.root, self.name)
        x, edge_index, y = net['attrb'], net['network'], net['group']
        x = torch.from_numpy(np.array(x).astype(float)).to(torch.float)
        # _, y = torch.max(torch.from_numpy(y).to(torch.int64), dim=1)
        edge_index= from_networkx(edge_index).edge_index

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])


def construct_ds(name):
    root = f"/home/xiaomeng/jupyter_base/BioTech/data/{name}"
    print(f'loading data from {root}')
    return DomainData(root, name=name)


def construct_src(args):
    return construct_ds(args.first_omni)


def construct_dst(args):
    return construct_ds(args.second_omni)


if __name__ == '__main__':
    data = construct_ds('ADT')
    data = construct_ds('SCT')
    print(1)

