import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from  torch import nn 
from torch.utils.data import DataLoader

from thermompnn.model.v2_model import TransferModelv2, TransferModelv2Siamese, TransferModelv2CQR
from thermompnn.trainer.trainer_utils import get_metrics


class TransferModelPLv2(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, cfg, train_dataset, val_dataset):
        super().__init__()
        if cfg.training.loss == 'pinball':
            self.model = TransferModelv2CQR(cfg)
        else:
            self.model = TransferModelv2(cfg)
        print(self.model)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        self.cfg = cfg
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.out = ['ddG']
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            
            for out in self.out:
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics(self.cfg.model.classifier).items():
                    self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        
        if self.cfg.model.subtract_mut and ('double' in self.cfg.data.mut_types):
            # do std fwd pass
            X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch
            fwd_preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            # modify seq and do reverse (mutant) pass
            backwd_preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_mutant_AAs, mut_wildtype_AAs, mut_ddGs, atom_mask)
            preds = fwd_preds - backwd_preds
        elif self.cfg.model.auxiliary_embedding == 'localESM':
            X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb = batch
            preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb)

        elif self.cfg.model.classifier:
            X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch
            preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            preds, mut_ddGs = preds.squeeze(-1), mut_ddGs.squeeze(-1)
            mut_ddGs = mut_ddGs.to(torch.int64)
            mse = F.cross_entropy(preds, mut_ddGs)
        else:
            X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch
            preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            if self.cfg.training.loss == 'pinball':
                mse = criterion(preds, mut_ddGs) # preds must be [B, 3] (upper/median/lower preds)
                preds = preds[..., 1] # use median for metric calculations
            else:
                mse = F.mse_loss(preds, mut_ddGs)

        for out in self.out:
            for metric in self.metrics[f"{prefix}_metrics"][out].values():
                try:
                    if self.cfg.model.classifier:
                        metric.update(torch.argmax(preds, dim=-1), mut_ddGs)
                    
                    else:
                        metric.update(torch.squeeze(preds), torch.squeeze(mut_ddGs))
                except IndexError:
                    continue

            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                try:
                    metric.compute()
                except ValueError:
                    continue
                self.log(f"{prefix}_{out}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                            batch_size=len(batch))
            
        if mse == 0.0:
            return None
        return mse

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        
        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.cfg.training.mpnn_learn_rate}]
        else: # fully frozen MPNN
            param_list = []

        if self.cfg.model.aggregation == 'mpnn':
            print('Loading double mutant aggregator params for optimizer!')
            param_list.append({"params": self.model.aggregator.parameters()})

        if self.cfg.model.dist:
            param_list.append({"params": self.model.dist_norm.parameters()})

        if self.cfg.model.lightattn:  # adding light attention parameters
            param_list.append({"params": self.model.light_attention.parameters()})

        if self.cfg.model.side_chain_module:
            print('Loading side chain encoder module params for optimizer!')
            param_list.append({"params": self.model.side_chain_features.parameters()})

        mlp_params = [
            {"params": self.model.ddg_out.parameters()}
            ]
        
        if self.cfg.training.loss == 'pinball':
            print('Loading upper/lower quantile params into optimizer!')
            param_list.append({"params": self.model.upper.parameters()})
            param_list.append({"params": self.model.lower.parameters()})

        
        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.cfg.training.learn_rate)

        if self.cfg.training.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': f'val_ddG_mse'
            }
        else:
            return opt

    def on_train_epoch_end(self):
        if (self.cfg.data.refresh_every > 0) and ((self.current_epoch % self.cfg.data.refresh_every) == 0):
            print('Reloading!')
            flag = True
        else:
            flag = False
        self.train_dataloader(flag=flag)
        self.val_dataloader(flag=flag)

    def train_dataloader(self, flag=False):
        if flag:
            self.train_dataset._refresh_dataset()

        esm = 'ESM' in self.cfg.model.auxiliary_embedding
        train_loader = DataLoader(self.train_dataset, 
                                    collate_fn=lambda b: tied_featurize_mut(b, side_chains=self.cfg.data.side_chains, esm=esm), 
                                    shuffle=self.cfg.training.shuffle, 
                                    num_workers=self.cfg.training.num_workers, 
                                    batch_size=self.cfg.training.batch_size)
        return train_loader

    def val_dataloader(self, flag=False):
        if flag:
            self.val_dataset._refresh_dataset()

        esm = 'ESM' in self.cfg.model.auxiliary_embedding
        val_loader = DataLoader(self.val_dataset, 
                                    collate_fn=lambda b: tied_featurize_mut(b, side_chains=self.cfg.data.side_chains, esm=esm), 
                                    shuffle=self.cfg.training.shuffle, 
                                    num_workers=self.cfg.training.num_workers, 
                                    batch_size=self.cfg.training.batch_size)
        return val_loader

class TransferModelPLv2Siamese(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, cfg, train_dataset, val_dataset):
        super().__init__()
        print('Multi-mutant siamese network enabled!')
        self.model = TransferModelv2Siamese(cfg)
        print(self.model)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.cfg = cfg
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.ALPHA = self.cfg.model.alpha # weight for avg MSE loss term
        self.BETA = self.cfg.model.beta # weight for sym loss term
        print('Relative loss weights:\nALPHA:\t%s\nBETA:\t%s' % (str(self.ALPHA), str(self.BETA)))
        self.out = ['ddG']
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            
            for out in self.out:
                self.metrics[split][out] = nn.ModuleDict()
                sym = self.cfg.model.aggregation == 'siamese'
                for name, metric in get_metrics(self.cfg.model.classifier, sym).items():
                    self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch
        # siamese net gives 2x ddGs that should match (one for each ordering)
        pred_ddG_A, pred_ddG_B = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)

        # symmetric loss function
        pred_ddG_avg = (pred_ddG_A + pred_ddG_B) / 2.
        pred_ddG_sym = torch.abs(pred_ddG_A - pred_ddG_B) / 2.
        # relative loss weights (hyperparameters to tune)

        mse = self.ALPHA * F.mse_loss(pred_ddG_avg, mut_ddGs) + self.BETA * torch.mean(pred_ddG_sym)

        for out in self.out:
            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                try:
                    if "sym" in name:
                        zeros = torch.zeros_like(pred_ddG_sym)
                        metric.update(torch.squeeze(pred_ddG_sym), torch.squeeze(zeros))
                    else:
                        metric.update(torch.squeeze(pred_ddG_avg), torch.squeeze(mut_ddGs))
                except IndexError:
                    continue

            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                try:
                    metric.compute()
                except ValueError:
                    continue
                self.log(f"{prefix}_{out}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                            batch_size=len(batch))
            
        if mse == 0.0:
            return None
        return mse

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        
        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.cfg.training.mpnn_learn_rate}]
        else: # fully frozen MPNN
            param_list = []

        if self.cfg.model.lightattn:  # adding light attention parameters
            param_list.append({"params": self.model.light_attention.parameters()})

        mlp_params = [
            {"params": self.model.ddg_out.parameters()}
            ]
        
        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.cfg.training.learn_rate)

        if self.cfg.training.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': f'val_ddG_mse'
            }
        else:
            return opt

    # def on_train_epoch_end(self):
    #     if (self.cfg.data.refresh_every > 0) and ((self.current_epoch % self.cfg.data.refresh_every) == 0):
    #         print('Reloading!')
    #         flag = True
    #     else:
    #         print('Not reloading dataloader!)
    #         flag = False
    #     self.train_dataloader(flag=flag)
    #     self.val_dataloader(flag=flag)

    # def train_dataloader(self, flag=False):
    #     if flag:
    #         self.train_dataset._refresh_dataset()

    #     esm = 'ESM' in self.cfg.model.auxiliary_embedding
    #     train_loader = DataLoader(self.train_dataset, 
    #                                 collate_fn=lambda b: tied_featurize_mut(b, side_chains=self.cfg.data.side_chains, esm=esm), 
    #                                 shuffle=self.cfg.training.shuffle, 
    #                                 num_workers=self.cfg.training.num_workers, 
    #                                 batch_size=self.cfg.training.batch_size)
    #     return train_loader

    # def val_dataloader(self, flag=False):
    #     if flag:
    #         self.val_dataset._refresh_dataset()

    #     esm = 'ESM' in self.cfg.model.auxiliary_embedding
    #     val_loader = DataLoader(self.val_dataset, 
    #                                 collate_fn=lambda b: tied_featurize_mut(b, side_chains=self.cfg.data.side_chains, esm=esm), 
    #                                 shuffle=self.cfg.training.shuffle, 
    #                                 num_workers=self.cfg.training.num_workers, 
    #                                 batch_size=self.cfg.training.batch_size)
    #     return val_loader


def criterion(input, target):

    ## Quantile Loss Levels (quantiles)
    q1 = 0.05
    q2 = 0.5
    q3 = 0.95
    
    ## Keras quantile loss, https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/    
    #e = y_p-y    
    #return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

    ## Quantile Loss
    ## for q1, q2, q3
    e1 = input[:,0:1] - target # !!! if input[:,0]  -> shape = (1000,)
    e2 = input[:,1:2] - target
    e3 = input[:,2:3] - target
    eq1 = torch.max(q1*e1, (q1-1)*e1)
    eq2 = torch.max(q2*e2, (q2-1)*e2)
    eq3 = torch.max(q3*e3, (q3-1)*e3)
    
    #eq1 = torch.max(0.05*e1, (0.05-1)*e1)
    #eq2 = torch.max(0.5*e2, (0.5-1)*e2)
    #eq3 = torch.max(0.95*e3, (0.95-1)*e3)
    loss = (eq1 + eq2 + eq3).mean()

    return loss


# class PinballLoss(nn.Module):
#     """ Pinball loss function
#     For some reason, torchlightning doesn't handle this properly - idk why - maybe it should be a _Loss, not nn.Module
#     From: https://github.com/yromano/cqr/blob/master/cqr/torch_models.py
#     """
#     def __init__(self, quantiles):
#         """ Initialize

#         Parameters
#         ----------
#         quantiles : pytorch vector of quantile levels, each in the range (0,1)


#         """
#         super().__init__()
#         self.quantiles = quantiles

#     def forward(self, preds, target):
#         """ Compute the pinball loss

#         Parameters
#         ----------
#         preds : pytorch tensor of estimated labels (n)
#         target : pytorch tensor of true labels (n)

#         Returns
#         -------
#         loss : cost function value

#         """
#         assert not target.requires_grad
#         assert preds.size(0) == target.size(0)
#         losses = []

#         for i, q in enumerate(self.quantiles):
#             errors = target - preds[:, i]
#             losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

#         loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
#         return loss