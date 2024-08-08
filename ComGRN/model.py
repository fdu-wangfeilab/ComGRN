import copy
import itertools
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



from tqdm import tqdm

from .network import MLP, NotearsDomain
from .utils import get_optimizer, kl_loss, squared_loss




class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    transforms = {}

    def __init__(self, input_dim, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.input_dim = input_dim
        self.num_domains = num_domains
        self.hparams = hparams

    def update(self, x, y, **kwargs):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def forward(self, x):
        return self.predict(x)

    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        clone.optimizer = self.new_optimizer(clone.network.parameters())
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone
    
    
    
class ComGRN(Algorithm):
    """
    DAG domain generalization methods

    """
    def __init__(self, input_dim, num_domains, hparams):
        super(ComGRN, self).__init__(input_dim,
                                   num_domains,
                                   hparams)
        
        # parameters
        self.n_domain = num_domains
        self.lambda_rec = self.hparams.lambda_rec
        self.lambda_kl1 = self.hparams.lambda_kl1
        self.lambda_kl2 = self.hparams.lambda_kl2
        self.lambda_clf = self.hparams.lambda_clf
        self.lambda_align = self.hparams.lambda_align
        self.lambda_recon_croase = self.hparams.lambda_recon_croase if hasattr(self.hparams, 'lambda_recon_croase') \
                                    else self.hparams.lambda_rec * 0.1
        
        self.rho_max = self.hparams.rho_max
        self.alpha = self.hparams.alpha
        self.rho = self.hparams.rho
        self._h_val = np.inf
        self.adv_step = self.hparams.adv_step

        # network
        self.feature_encoder = MLP(input_dim, input_dim * 2, self.hparams)
        self.domain_encoder = MLP(num_domains, hparams.num_hidden * 2, self.hparams)

        self.dag_mlp = NotearsDomain(input_dim, hparams.num_hidden, nonlinear=self.hparams.nonlinear)
        self.dag_mlp.weight_pos.data[1:, 0].fill_(0.0)
        self.dag_mlp.weight_neg.data[1:, 0].fill_(0.0)

        self.disc = MLP(input_dim, num_domains, self.hparams)

        self.network = nn.Sequential(self.feature_encoder, self.domain_encoder, self.dag_mlp)
        
        
        
        self.optim = get_optimizer(
            hparams.optimizer,
            [{"params": self.network.parameters()}],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        self.disc_optim = get_optimizer(
            hparams.optimizer,
            [{"params": self.disc.parameters()}],
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def fit(self, 
            dataloader, 
            num_epoch, 
            device,
            progress_bar = False, 
            tune_flag = False):
        self.to(device)
        step = 0
        start_time = time.time()
        for epoch in range(num_epoch):
            if progress_bar:
                dataloader = tqdm(dataloader)
            train_loss, step = self.train_epoch(dataloader, step, epoch, device, tune_flag)

        end_time = time.time()
        print("cost time: ", end_time - start_time)

    def train_epoch(self, dataloader, step, epoch, device):
        self.to(device)
        self.train()
        
        datasize = 0
        early_stop_loss = 0
        
        for x, d, idx in dataloader:
            exprs = x.float().to(device)
            batch = d.long().to(device)
            datasize += batch.shape[0]
            step += 1
            
            loss = self.update(exprs, batch, step)
            
            early_stop_loss += loss['l2']
        
        if epoch % 10 == 0:
            print(epoch, step, loss)
            
        return loss, step
    
    
    
    def update(self, x, batch, step):
        
        # train a good disc
        for i in range(self.adv_step): 
            z = self.feature_encoder(x)
            mu, log_var = torch.chunk(z, 2, dim=-1)

            loss_disc = F.cross_entropy(self.disc(mu), batch)
            self.disc_optim.zero_grad()
            loss_disc.backward()
            self.disc_optim.step()
        
        # train model
        z = self.feature_encoder(x)
        mu, log_var = torch.chunk(z, 2, dim=-1)
        batch_oh = F.one_hot(batch, num_classes=self.n_domain).to(torch.float)

        domain = self.domain_encoder(batch_oh)
        d_mu, d_log_var = torch.chunk(domain, 2, dim=1)
        domain = self.reparameterize(d_mu, d_log_var)

        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=d_mu)
        rec_x = self.reparameterize(rec_x_mu, log_var)
        clean_x = self.reparameterize(clean_x_mu, log_var)    

        rec_u = self.dag_mlp.mask_feature(x=x).squeeze(-1)

        # reconstruction loss
        loss_rec_1 = squared_loss(rec_x, x) * self.lambda_rec
        loss_rec_2 = squared_loss(rec_u, x) * self.lambda_recon_croase
        loss_rec = loss_rec_1 + loss_rec_2
        
        # kl loss
        loss_kl_1 = kl_loss(mu, log_var) * self.lambda_kl1
        loss_kl_2 = kl_loss(d_mu, d_log_var) * self.lambda_kl2
        loss_kl = loss_kl_1 + loss_kl_2
        
        # dag loss
        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l2_reg = 0.5 * 0.01 * self.dag_mlp.l2_reg() # parameter # 0.5
        l1_reg = 0.01 * self.dag_mlp.w_l1_reg() # parameter 0.01

        loss_dag =  penalty + l2_reg 

        loss_g_disc = -F.cross_entropy(self.disc(mu), batch) * self.lambda_align   
        
        if step >= self.hparams["dag_anneal_steps"]:
            loss = loss_rec + loss_dag + loss_g_disc + loss_kl
        else:
            loss = loss_rec + loss_g_disc + loss_kl
            loss_dag = torch.tensor(0.0)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return {"loss": loss.item(),
                "disc": loss_g_disc.item(),
                "recon1": loss_rec_1.item(),
                "recon2": loss_rec_2.item(),
                "kl1": loss_kl_1.item(),
                "kl2": loss_kl_2.item(),
                "dag": loss_dag.item(),
                "penalty": penalty.item(),
                "l1": l1_reg.item(),
                "l2": l2_reg.item()}
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def predict(self, x, batch):
        z = self.feature_encoder(x)
        mu, log_var = torch.chunk(z, 2, dim=-1)

        batch_oh = F.one_hot(batch, num_classes=self.n_domain).to(torch.float)
        domain = self.domain_encoder(batch_oh)
        d_mu, d_log_var = torch.chunk(domain, 2, dim=1)
        
        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=d_mu)
        rec_x = self.reparameterize(rec_x_mu, log_var)
        clean_x = self.reparameterize(clean_x_mu, log_var)
        
        return rec_x_mu, clean_x_mu, mu, domain

    def predict_clean_data(self, x):
        self.eval()
        z = self.feature_encoder(x)
        mu, log_var = torch.chunk(z, 2, dim=-1)
        
        batch = torch.zeros((x.shape[0]), dtype=torch.long).to(x.device)
        batch_oh = F.one_hot(batch, num_classes=self.n_domain).to(torch.float)
        domain = self.domain_encoder(batch_oh)
        d_mu, d_log_var = torch.chunk(domain, 2, dim=1)
        domain = self.reparameterize(d_mu, d_log_var)
        
        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, inverse=True, y=d_mu)
        clean_x = self.reparameterize(clean_x_mu, log_var)
        return clean_x_mu

    def evaluate(self, dataloader, data, device):
        self.eval()
        num = data.shape[0]
        dim = data.shape[1]
        denoises = np.zeros((num, dim))
        recons = np.zeros((num, dim))
        latents = np.zeros((num, dim))
        # domains = np.zeros((num, dim))
        ds = np.zeros((data.shape[0],))
        for (x, d, idx) in tqdm(dataloader, total=len(dataloader)):
            x, d = x.float().to(device), d.long().to(device)
            recon_x, clean_x, z, domain = self.predict(x, d)

            recons[idx] = recon_x.detach().cpu().numpy()
            denoises[idx] = clean_x.detach().cpu().numpy()
            latents[idx] = z.detach().cpu().numpy()
            # domains[idx] = domain.detach().cpu().numpy()
            
            ds[idx] = d.detach().cpu().numpy()
  
        data.obsm['X_denoised'] = denoises
        data.obsm['X_recon'] = recons
        data.obsm['X_latent'] = latents
        return data
    
    def transfer(self, x, batch, domain_id):
        self.eval()
        z = self.feature_encoder(x)
        mu, log_var = torch.chunk(z, 2, dim=-1)

        batch = torch.full_like(batch, domain_id, dtype=torch.long).to(x.device)
        batch_oh = F.one_hot(batch, num_classes=self.n_domain).to(torch.float)
        domain = self.domain_encoder(batch_oh)
        d_mu, d_log_var = torch.chunk(domain, 2, dim=1)
        domain = self.reparameterize(d_mu, d_log_var)

        rec_x_mu, clean_x_mu = self.dag_mlp(x=mu, y=d_mu)
        rec_x = self.reparameterize(rec_x_mu, log_var)
        clean_x = self.reparameterize(clean_x_mu, log_var)
        
        return rec_x_mu 
    
    def data_transform(self, dataloader, data, domain_id, device):
        num = data.shape[0]
        dim = data.shape[1]
        trans = np.zeros((num, dim))
        for (x, d, idx) in tqdm(dataloader, total=len(dataloader)):
            x, d = x.float().to(device), d.long().to(device)
            recon_x = self.transfer(x, d, domain_id)
            trans[idx] = recon_x.detach().cpu().numpy()
        data.obsm['X_trans'] = trans
        return data
    
    def data_denoise(self, dataloader, data, device):
        num = data.shape[0]
        dim = data.shape[1]
        denoised = np.zeros((num, dim))
        for (x, d, idx) in tqdm(dataloader, total=len(dataloader)):
            x, d = x.float().to(device), d.long().to(device)
            clean_x = self.predict_clean_data(x)
            denoised[idx] = clean_x.detach().cpu().numpy()
        data.obsm['X_denoised'] = denoised
        return data