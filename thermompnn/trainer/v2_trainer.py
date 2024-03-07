import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from  torch import nn 

from thermompnn.model.v2_model import TransferModelv2
from thermompnn.trainer.trainer_utils import get_metrics


class TransferModelPLv2(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, cfg):
        super().__init__()
        self.model = TransferModelv2(cfg)

        self.cfg = cfg
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.out = ['ddG']
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            
            for out in self.out:
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics().items():
                    self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):

        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch
        if self.cfg.model.subtract_mut and 'double' in self.cfg.data.mut_types:
            # do std fwd pass
            fwd_preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            # modify seq and do reverse (mutant) pass
            backwd_preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_mutant_AAs, mut_wildtype_AAs, mut_ddGs, atom_mask)
            preds = fwd_preds - backwd_preds
        else:
            preds, _ = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
        mse = F.mse_loss(preds, mut_ddGs)

        for out in self.out:
            for metric in self.metrics[f"{prefix}_metrics"][out].values():
                metric.update(torch.squeeze(preds), torch.squeeze(mut_ddGs))

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
            print('Loading light attention layer params for optimizer!')
            param_list.append({"params": self.model.light_attention.parameters()})

        if self.cfg.model.side_chain_module:
            print('Loading side chain encoder module params for optimizer!')
            param_list.append({"params": self.model.side_chain_features.parameters()})

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
