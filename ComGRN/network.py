# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from torch.autograd import Variable

class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Sequential(
                    nn.Linear(n_inputs, hparams.mlp_width),
                    nn.LayerNorm(hparams.mlp_width),
                    nn.LeakyReLU(hparams.mlp_ac),
                    nn.Dropout(hparams.mlp_dropout)
                )
        self.hiddens = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hparams.mlp_width, hparams.mlp_width),
                    nn.LayerNorm(hparams.mlp_width),
                    nn.LeakyReLU(hparams.mlp_ac),
                    nn.Dropout(hparams.mlp_dropout)
                )
                for _ in range(hparams.mlp_depth - 2)
            ]
        )
        self.output = nn.Linear(hparams.mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        for hidden in self.hiddens:
            x = hidden(x)
        x = self.output(x)
        return x

class LocallyConnected(nn.Module):
    """
    Local linear layer, i.e., Conv1dLocal() with filter size 1.

    Args:
        num_linear: num of local linear layers
        in_features: m1
        out_features: m2
        bias: whether to include bias

    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]

    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """
    def __init__(self, num_linear, in_features, out_features, bias=True):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                in_features,
                                                out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.in_features
        bound = np.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        out = torch.matmul(inputs.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            out += self.bias
        return out

    def extra_repr(self):
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.in_features, self.out_features,
            self.bias is not None
        )

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >=2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        self.d = dims[0]
        self.register_buffer("_identity", torch.eye(d))
        # fc1: variable spliting for l1 ref: <http://arxiv.org/abs/1909.13189>
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        # specific bounds for customize optimizer
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(nn.Sigmoid())
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.Sequential(*layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x):
        # [n, d] -> [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        x = self.fc2(x)
        x = x.squeeze(dim=2)
        return x

    def h_func(self):
        """
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG
        """
        d = self.dims[0]
        # [j * m1, i]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        # h = torch.trace(torch.matrix_exp(A)) - d
        # A different formulation, slightly faster at the cost of numerical stability
        M = self._identity + A / self.d
        E = torch.matrix_power(M, self.d - 1)
        h = (E.t() * M).sum() - self.d
        return h

    def l2_reg(self):
        """
        Take 2-norm-squared of all parameters
        """
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        reg += torch.sum(fc1_weight ** 2)

        for fc in self.fc2:
            if hasattr(fc, 'weight'):
                reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """
        Take l1 norm of fc1 weight
        """
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:
        """
        Get W from fc1 weight, take 2-norm over m1 dim
        """
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        W = torch.sqrt(A)
        W = W.cpu().detach().numpy()
        return W

    @torch.no_grad()
    def fc1_to_p_sub(self) -> torch.Tensor:
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight
        fc1_weight = fc1_weight.view(d, -1, d)
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()
        P_sub = torch.inverse(self._identity - A)
        return P_sub

class Notears(nn.Module):
    def __init__(self, dims):
        super(Notears, self).__init__()
        self.dims = dims
        self.weight_pos = nn.Parameter(torch.rand(dims, dims) * 0.001)
        self.weight_neg = nn.Parameter(torch.rand(dims, dims) * 0.001)

    def adj_inv(self):
        W = self.adj()
        adj_normalized = torch.Tensor(np.eye(self.dims)).to(W) - (W.transpose(0, 1))
        adj_inv = torch.inverse(adj_normalized)
        return adj_inv

    def adj(self):
        return self.weight_pos - self.weight_neg
    
    def h_func(self):
        W = self.adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.mean(self.weight_pos + self.weight_neg)
        return reg
    
    def l2_reg(self):
        reg = 0.
        weight = self.weight_pos - self.weight_neg
        reg += torch.sum(weight ** 2)
        return reg

    def forward(self, x):
        W_inv = self.adj_inv() 
        recon_x = x @ W_inv
        return recon_x

class NotearsClassifier(nn.Module):
    def __init__(self, dims, num_classes):
        super(NotearsClassifier, self).__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.weight_pos = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.weight_neg = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_classes

    def _adj(self):
        W = self.weight_pos - self.weight_neg
        adj_normalized = torch.Tensor(np.eye(self.dims + 1)).to(W) - (W.transpose(0, 1))
        adj_inv = torch.inverse(adj_normalized)
        return adj_inv
    
    def _adj(self):
        return self.weight_pos - self.weight_neg

    def _adj_sub(self):
        W = self._adj()
        return torch.matrix_exp(W * W)

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.mean(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x, y=None):
        W = self._adj()
        W_sub = self._adj_sub()
        if y is not None:
            x_aug = torch.cat((x, y.unsqueeze(1)), dim=1)
            M = x_aug @ W
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0)
            # reconstruct variables, classification logits
            return M[:, :self.dims], masked_x
        else:
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0).detach()
            return masked_x

    def mask_feature(self, x):
        W_sub = self._adj_sub()
        mask = W_sub[:self.dims, -1].unsqueeze(0).detach()
        return x * mask

    @torch.no_grad()
    def projection(self):
        self.weight_pos.data.clamp_(0, None)
        self.weight_neg.data.clamp_(0, None)
        self.weight_pos.data.fill_diagonal_(0)
        self.weight_neg.data.fill_diagonal_(0)

    @torch.no_grad()
    def masked_ratio(self):
        W = self._adj()
        return torch.norm(W[:self.dims, -1], p=0)
    
class NotearsDomain(nn.Module):
    def __init__(self, dims, num_domains, nonlinear=False):
        super(NotearsDomain, self).__init__()
        self.dims = dims
        self.num_domains = num_domains
        self.weight_pos = nn.Parameter(torch.rand(dims + num_domains, dims + num_domains) * 0.001)
        self.weight_neg = nn.Parameter(torch.rand(dims + num_domains, dims + num_domains) * 0.001)
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_domains
        self.prelu = nn.PReLU()
        self.nonlinear = nonlinear
        
    def adj_t(self, clean_flg=False):
        W = self.adj(clean_flg=clean_flg)
        num = W.shape[0] 
        adj_normalized = torch.Tensor(np.eye(num)).to(W) - (W.transpose(0, 1))
        return adj_normalized
    
    def adj_inv(self, clean_flg=False):
        adj_normalized = self.adj_t(clean_flg=clean_flg)
        adj_inv = torch.inverse(adj_normalized)
        return adj_inv

    def adj(self, clean_flg):
        if clean_flg:
            W = self.weight_pos[self.num_domains:, self.num_domains:] - self.weight_neg[self.num_domains:, self.num_domains:]
        else:
            W = self.weight_pos - self.weight_neg
        num = W.shape[0] 
        mask = Variable(torch.from_numpy(np.ones(num) - np.eye(num)).float(),
                        requires_grad=False).to(W)
        W = W * mask
        return W
    
    def adj_sub(self, clean_flg):
        W = self.adj(clean_flg)
        return torch.matrix_exp(W * W)
    
    def h_func(self):
        W = self.adj(clean_flg=False)
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - self.num_domains
        return h

    def w_l1_reg(self):
        reg = torch.mean(self.weight_pos + self.weight_neg)
        return reg
    
    def l2_reg(self):
        reg = 0.
        weight = self.weight_pos - self.weight_neg
        reg += torch.sum(weight ** 2)
        return reg

    def forward(self, x, y=None):
        W_inv = self.adj_inv(clean_flg=False)
        W_inv_clean = self.adj_inv(clean_flg=True) 
        if y is not None:
            x_aug = torch.cat((y, x), dim=1)
            M = x_aug @ W_inv 
            clean_x = x @ W_inv_clean
            if self.nonlinear:
                return self.prelu(M[:, self.num_domains:]), self.prelu(clean_x)
            else:
                return M[:, self.num_domains:], clean_x
        else:
            clean_x = x @ W_inv_clean
            if self.nonlinear:
                return self.prelu(M[:, self.num_domains:]), self.prelu(clean_x)
            else:
                return M[:, self.num_domains:], clean_x

        
    def mask_feature(self, x):
        W = self.adj(clean_flg=True)
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(W.t(), x)
        if self.nonlinear:
            return self.prelu(x)
        else:
            return x
        


