import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Metric
import pytorch_lightning as pl

from thermompnn.model.siamese_model import TransferModelSiamese
from thermompnn.model.trainer_utils import get_metrics

def custom_log_cosh_loss(D, y ,S):
    def _log_cosh(D, y, S):
        # logcosh minimum is at 0, so this minimizes true/pred difference
        # S is the avg ddg which should also be minimized down to 0
        return torch.log(torch.cosh(D - y)) + torch.abs(S)
    return torch.mean(_log_cosh(D, y, S))

class CustomLogCoshLoss(torch.nn.Module):
    """Regression loss fxn used in ACDC-NN (NOT standard log-cosh loss)"""
    def __init__(self):
        super().__init__()

    def forward(self, D, y, S):
        # D is target_ddg, y is true_ddg, S is avg_ddg
        return custom_log_cosh_loss(D, y, S)


class CustomLogCoshMetric(Metric):
    def __init__(self):
        super().__init__()
        # define tracked variables
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, D, y, S):
        # add next metric batch to total and count number of elements
        assert D.shape == y.shape == S.shape
        
        self.total_loss += torch.sum(custom_log_cosh_loss(D, y, S))
        self.total += 1

    def compute(self):
        # get avg of metric
        return self.total_loss.float() / self.total
    
def log_cosh_only(D, y ,S):
    def _log_cosh(D, y, S):
        # logcosh minimum is at 0, so this minimizes true/pred difference
        # S is the avg ddg which should also be minimized down to 0
        return torch.log(torch.cosh(D - y))
    return torch.mean(_log_cosh(D, y, S))

class LogCoshOnly(Metric):
    """Only first loss term (for metric tracking)"""
    def __init__(self):
        super().__init__()
        # define tracked variables
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, D, y, S):
        # add next metric batch to total and count number of elements
        assert D.shape == y.shape == S.shape
        
        self.total_loss += torch.sum(log_cosh_only(D, y, S))
        self.total += 1

    def compute(self):
        # get avg of metric
        return self.total_loss.float() / self.total

def abs_loss_only(D, y ,S):
    def _log_cosh(D, y, S):
        # logcosh minimum is at 0, so this minimizes true/pred difference
        # S is the avg ddg which should also be minimized down to 0
        return torch.abs(S)
    return torch.mean(_log_cosh(D, y, S))

class AbsLossOnly(Metric):
    """Only first loss term (for metric tracking)"""
    def __init__(self):
        super().__init__()
        # define tracked variables
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, D, y, S):
        # add next metric batch to total and count number of elements
        assert D.shape == y.shape == S.shape
        
        self.total_loss += torch.sum(abs_loss_only(D, y, S))
        self.total += 1

    def compute(self):
        # get avg of metric
        return self.total_loss.float() / self.total
    

class TransferModelSiamesePL(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, cfg):
        super().__init__()
        self.model = TransferModelSiamese(cfg)

        self.cfg = cfg
        self.learn_rate = cfg.training.learn_rate
        self.mpnn_learn_rate = cfg.training.mpnn_learn_rate if 'mpnn_learn_rate' in cfg.training else None
        self.lr_schedule = cfg.training.lr_schedule if 'lr_schedule' in cfg.training else False
        self.avg_multiplier = float(self.cfg.training.avg_multiplier) if 'avg_multiplier' in self.cfg.training else 1.0
        print('AVG MULTIPLIER:', self.avg_multiplier)
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.loss = cfg.training.loss_fxn if 'loss_fxn' in cfg.training else 'mse'
        if self.loss != 'mse':
            print('Using custom COSH loss')
            self.log_cosh = CustomLogCoshLoss()

        # set up metrics dictionary
        self.extra_metrics = nn.ModuleDict()
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "ddG"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics().items():
                self.metrics[split][out][name] = metric
            
            # TODO initialize custom metrics
            # if self.loss != 'mse':
            self.extra_metrics[split] = nn.ModuleDict()
            self.extra_metrics[split][out] = nn.ModuleDict()
            self.extra_metrics[split]['ddG']['full_log_cosh_loss'] = CustomLogCoshMetric()
            self.extra_metrics[split]['ddG']['log_cosh_loss_only'] = LogCoshOnly()
            self.extra_metrics[split]['ddG']['abs_loss_only'] = AbsLossOnly()

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        
        assert len(batch) == 2
        ddg1, ddg2 = self(batch[0], batch[1])
        true_ddg = batch[0][-1]

        # should equal target ddg
        target_ddg = (ddg1 - ddg2) / 2
        
        # should equal 0
        avg_ddg = (ddg1 + ddg2) / 2
        
        if self.loss == 'mse': # just a lazy MSE type loss
            loss = F.mse_loss(target_ddg, true_ddg) + F.mse_loss(avg_ddg, torch.zeros_like(avg_ddg)) * self.avg_multiplier
        else: # acdc-nn custom loss
            D = target_ddg
            y = true_ddg
            S = avg_ddg
            loss = self.log_cosh(D, y, S)
        
        # # record training/validation metrics
        for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
            metric.update(torch.squeeze(target_ddg), torch.squeeze(true_ddg))
            
        # add extra metric (custom loss fxn)
        for extra_metric in self.extra_metrics[f"{prefix}_metrics"]["ddG"].values():
            D = target_ddg
            y = true_ddg
            S = avg_ddg
            extra_metric.update(D, y, S)
            
        on_step = False
        on_epoch = not on_step
        
        output = "ddG"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
        
        # extra metric logging
        for name, extra_metric in self.extra_metrics[f"{prefix}_metrics"][output].items():
            try:
                extra_metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", extra_metric, prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))    
        
        if loss == 0.0:
            return None
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        
        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.mpnn_learn_rate}]
        else: # fully frozen MPNN
            param_list = []

        if self.model.lightattn:  # adding light attention parameters
            param_list.append({"params": self.model.light_attention.parameters()})

        mlp_params = [
            {"params": self.model.ddg_out.parameters()}
            ]

        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

        if self.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': 'val_ddG_mse'
            }
        else:
            return opt

