import networkx
import scanpy
import torch
import numpy as np
import pandas
import pickle
from sklearn.neighbors import NearestNeighbors


def file2coo(file, opt_root, dataset_name, neighbors=8):
    feature = scanpy.read_h5ad(file)
    edge_dict = set()
    cell_type_l1 = feature.obs['celltype.l1']
    cell_type_l2 = feature.obs['celltype.l2']
    cell_type_l3 = feature.obs['celltype.l3']
    for index, l1 in enumerate(cell_type_l1):
        l2 = cell_type_l2[index]
        l3 = cell_type_l3[index]
        edge_dict.add(f'l1:{l1}=>l2:{l2}=>l3:{l3}')
    print(edge_dict)
    print(len(edge_dict))
    unique_set = set()
    for item in edge_dict:
        unique_set.add(item.split(':')[-1])
    assert len(unique_set) == len(edge_dict)

    cols = feature.var['features']
    adt_f = feature.X
    frame = pandas.DataFrame(np.concatenate((adt_f, np.array(cell_type_l3).reshape(-1, 1)), 1))
    _cols = []
    for i in cols:
        _cols.append(i)
    _cols.append('cls')
    frame.index = cell_type_l3.index
    frame.columns = _cols
    csv_path = opt_root + f'/{dataset_name}.csv'
    print(f'original feature output to {csv_path}')
    frame.to_csv(csv_path)

    neigh = NearestNeighbors(n_neighbors=8)
    neigh.fit(frame.values[:, 1:-1])
    adj_matrix = neigh.kneighbors_graph(frame.values[:, 1:-1])
    adj_path = opt_root + f'/{dataset_name}.adj.top{neighbors}.npy'
    np.save(adj_path, adj_matrix)
    print(f'original adj matrix with top-{neighbors} to {adj_path}')

    graph_path = opt_root + f'/{dataset_name}.networkx.g'
    networkx_g = networkx.from_numpy_matrix(adj_matrix)
    print(f'network format graph output to {graph_path}')
    with open(graph_path, 'wb') as f:
        pickle.dump(networkx_g, f)

    feat = torch.tensor(frame.values[:, 1:-1].astype(float))
    feat_path = opt_root + f'/{dataset_name}.feat.pt'
    print(f'tensor type feat output to {feat_path}')
    torch.save(feat, feat_path)

    adj_coo = networkx.to_scipy_sparse_matrix(networkx_g, format='coo', weight=None)
    adj_coo_path = opt_root + f'/{dataset_name}.adj.coo.top-{neighbors}.npy'
    print(f'adj coo format numpy file output to {adj_coo_path}')
    np.save(adj_coo_path, adj_coo)

    label = torch.tensor(frame.values[:, -1].astype(float))
    label_path = opt_root + f'/{dataset_name}.label.pt'
    print(f'save {dataset_name} label file output to {label_path}')
    torch.save(label, label_path)


