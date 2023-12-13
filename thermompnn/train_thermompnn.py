import sys
import wandb
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

from thermompnn.parsers import get_v1_dataset, get_v2_dataset, get_siamese_dataset
from thermompnn.trainer.v1_trainer import TransferModelPL
from thermompnn.trainer.v2_trainer import TransferModelPLv2
from thermompnn.trainer.siamese_trainer import TransferModelSiamesePL


def train(cfg):
    print('Configuration:\n', cfg)
    
    if 'project' in cfg:
        wandb.init(project=cfg.project, name=cfg.name)
    else:
        cfg.name = 'test'
    
    # params that need to be set before data/model initialization
    if cfg.training.batch_size is None:
        cfg.training.batch_size = 10000
    
    num_workers = cfg.training.num_workers if 'num_workers' in cfg.training else 0
    train_workers, val_workers = int(num_workers * 0.75), int(num_workers * 0.25)

    # pick which ThermoMPNN version to use (v1, v2, or siamese)
    print(f'Loading ThermoMPNN version {cfg.version}')
    if cfg.version.lower() == 'v1':
        # initialize datasets
        train_dataset, val_dataset = get_v1_dataset(cfg)
        train_loader = DataLoader(train_dataset, collate_fn=lambda x: x, shuffle=True, num_workers=train_workers)
        val_loader = DataLoader(val_dataset, collate_fn=lambda x: x, num_workers=val_workers, shuffle=False)

        # load transfer learning model and trainer
        model_pl = TransferModelPL(cfg)

    elif cfg.version.lower() == 'v2':
        train_dataset, val_dataset = get_v2_dataset(cfg)
        model_pl = TransferModelPLv2(cfg)
        train_loader = DataLoader(train_dataset, collate_fn=None, shuffle=True, num_workers=train_workers, batch_size=None)
        val_loader = DataLoader(val_dataset, collate_fn=None, num_workers=val_workers, batch_size=None, shuffle=False)
        
    elif cfg.version.lower() == 'siamese':
        train_dataset, val_dataset = get_siamese_dataset(cfg)
        model_pl = TransferModelSiamesePL(cfg)
        train_loader = DataLoader(train_dataset, collate_fn=None, shuffle=True, num_workers=train_workers, batch_size=None)
        val_loader = DataLoader(val_dataset, collate_fn=None, num_workers=val_workers, batch_size=None, shuffle=False)

    else:
        raise ValueError('Invalid ThermoMPNN version! Must be v1, v2, or siamese')
    
    # additional params, logging, checkpoints for training
    max_ep = cfg.training.epochs if 'epochs' in cfg.training else 100
    batch_fraction = cfg.training.batch_fraction if 'batch_fraction' in cfg.training else 1.0

    filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = 'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath='checkpoints', filename=filename)
    logger = WandbLogger(project=cfg.project, name="test", log_model=False) if 'project' in cfg else None
    
    # start training
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep,
                         accelerator=cfg.platform.accel, devices=1, limit_train_batches=batch_fraction, limit_val_batches=batch_fraction)
    trainer.fit(model_pl, train_loader, val_loader)


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) != 3:
        raise ValueError("Need to specify exactly two config files (config.yaml and local.yaml)")
    
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))
    train(cfg)
