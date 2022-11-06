from torch_geometric.nn import VGAE

from model.dataset import construct_src, construct_dst, construct_ds
import torch.nn as nn
import torch
# import dance.utils.metrics as metrics
from model.utils.metrics import labeled_clustering_evaluate


class Wrapper(object):
    def __init__(self, encoder, decoder, ds1, ds2, args):
        self.dataset1 = ds1.data
        self.dataset2 = ds2.data
        self.label = self.dataset1.y
        self.feature = torch.cat((torch.tensor(self.dataset1.x.float()), torch.tensor(self.dataset2.x.float())), dim = -1)
        self.vgae = VGAE(encoder, decoder)
        self.optimizer = torch.optim.AdamW([{'params': self.vgae.parameters()}])
        self.mse = nn.MSELoss()
        self.l1_w = args.l1_w
        self.l2_w = args.l2_w
        self.l3_w = args.l3_w
        self.l4_w = args.l4_w
        self.epoch = args.epoches
        self.eval_epoch = args.report_epoch
        self.args = args
        self.clusters = len(set(self.dataset1.y.tolist()))

    def fit(self):
        for epoch in range(self.epoch):
            self.vgae.train()
            self.optimizer.zero_grad()
            y_1 = self.dataset1.x.float()
            y_2 = self.dataset2.x.float()

            y_hat_1 = self.encode(self.feature, self.dataset1)
            y_hat_2 = self.encode(self.feature, self.dataset2)
            # feature recover loss
            loss1 = self.mse(y_hat_1, y_1)
            loss2 = self.mse(y_hat_2, y_2)
            # reconstruction loss
            loss3 = self.recon_loss(y_hat_1, self.dataset1)
            loss4 = self.recon_loss(y_hat_2, self.dataset2)
            # cross loss
            loss5 = self.mse(y_hat_2, y_1)
            loss6 = self.mse(y_hat_1, y_2)
            # use source classifier loss:
            loss = self.l1_w * loss1 + self.l1_w * loss2 + self.l3_w * loss3 + self.l4_w * loss4 + self.l2_w * loss5 + self.l2_w * loss6
            print(f'within loss:{loss1.round(2)}, {loss2.round(2)}', f'recon loss: {loss3.round(2)}, {loss4.round(2)}',
                  f'cross loss: {loss5.round(2)}, {loss6.round(2)}')
            print(f'weighted loss:{loss}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch % self.eval_epoch == 0:
                self.eval(y_hat_1, y_hat_2)

    def reset_parameter(self):
        pass

    def to(self, device):
        self.args.device = device
        self.model = self.vgae.to(device)
        self.dataset2 = self.dataset1.to(device)
        self.dataset2 = self.dataset2.to(device)
        self.feature = self.feature.to(device)
        return self

    def load(self, path, map_location=None):
        if map_location is not None:
            self.model.load_state_dict(torch.load(path, map_location=map_location))
        else:
            self.model.load_state_dict(torch.load(path))

    def encode(self, x, data, mask=None):
        encoded_output = self.vgae.encode(x, data.edge_index)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output

    def decode(self, z, data, mask=None):
        decoded_output = self.vgae.decode(z, data.edge_index)
        if mask is not None:
            decoded_output = decoded_output[mask]
        return decoded_output

    def recon_loss(self, z, data, mask=None):
        recon_loss = self.vgae.recon_loss(z, data.edge_index)
        return recon_loss

    def eval(self, y_hat_1=None, y_hat_2=None):
        if y_hat_1 == None:
            y_hat_1 = self.encode(self.feature, self.dataset1)
        if y_hat_2 == None:
            y_hat_2 = self.encode(self.feature, self.dataset2)
        NMI_score, ARI_score = labeled_clustering_evaluate(torch.cat((y_hat_1, y_hat_2), -1),
                                                           label=self.dataset1.y, cluster=self.clusters)
        return NMI_score, ARI_score
