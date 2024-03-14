import sys
import wandb
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
from functools import partial

from thermompnn.parsers import get_v1_dataset, get_v2_dataset, get_siamese_dataset
from thermompnn.trainer.v1_trainer import TransferModelPL
from thermompnn.trainer.v2_trainer import TransferModelPLv2
from thermompnn.trainer.siamese_trainer import TransferModelSiamesePL


def parse_cfg(cfg):
    """
    Parse CFG and set default arguments as needed
    """

    cfg.project = cfg.get('project', None)
    cfg.name = cfg.get('name', 'test')

    # data config
    cfg.data = cfg.get('data', {})
    cfg.data.mut_types = cfg.data.get('mut_types', ['single'])
    cfg.data.splits = cfg.data.get('splits', ['train', 'val'])
    cfg.data.side_chains = cfg.data.get('side_chains', False)

    # training config
    cfg.training = cfg.get('training', {})
    cfg.training.num_workers = cfg.training.get('num_workers', 0)
    cfg.training.batch_size = cfg.training.get('batch_size', 256)
    cfg.training.epochs = cfg.training.get('epochs', 100)
    cfg.training.batch_fraction = cfg.training.get('batch_fraction', 1.0)
    cfg.training.shuffle = cfg.training.get('shuffle', True)

    cfg.training.learn_rate = cfg.training.get('learn_rate', 0.0001)
    cfg.training.mpnn_learn_rate = cfg.training.get('mpnn_learn_rate', None)
    cfg.training.lr_schedule = cfg.training.get('lr_schedule', True)

    # model config
    cfg.model = cfg.get('model', {})
    cfg.model.hidden_dims = cfg.model.get('hidden_dims', [64, 32])
    cfg.model.subtract_mut = cfg.model.get('subtract_mut', True)
    cfg.model.single_target = cfg.model.get('single_target', False)
    cfg.model.num_final_layers = cfg.model.get('num_final_layers', 2)
    cfg.model.freeze_weights = cfg.model.get('freeze_weights', True)
    cfg.model.load_pretrained = cfg.model.get('load_pretrained', True)
    cfg.model.lightattn = cfg.model.get('lightattn', True)
    cfg.model.mutant_embedding = cfg.model.get('mutant_embedding', False)
    
    # global/new featurization options
    cfg.model.auxiliary_embedding = cfg.model.get('auxiliary_embedding', '')

    # double mutant model options
    cfg.model.dist = cfg.model.get('dist', False)
    cfg.model.edges = cfg.model.get('edges', False)
    cfg.model.aggregation = cfg.model.get('aggregation', None)
    cfg.model.dropout = cfg.model.get('dropout', None)

    # side chain model options
    cfg.model.side_chain_module = cfg.model.get('side_chain_module', False)
    cfg.model.action_centers = cfg.model.get('action_centers', None)

    return cfg


def train(cfg):
    print('Configuration:\n', cfg)

    cfg = parse_cfg(cfg)

    if cfg.project is not None:
        wandb.init(project=cfg.project, name=cfg.name)

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
        from thermompnn.datasets.v2_datasets import tied_featurize_mut

        esm = 'ESM' in cfg.model.auxiliary_embedding

        train_loader = DataLoader(train_dataset, 
                                    collate_fn=lambda b: tied_featurize_mut(b, side_chains=cfg.data.side_chains, esm=esm), 
                                    shuffle=cfg.training.shuffle, 
                                    num_workers=cfg.training.num_workers, 
                                    batch_size=cfg.training.batch_size)
        val_loader = DataLoader(val_dataset, 
                                    collate_fn=lambda b: tied_featurize_mut(b, side_chains=cfg.data.side_chains, esm=esm), 
                                    shuffle=False, 
                                    num_workers=cfg.training.num_workers, 
                                    batch_size=cfg.training.batch_size)

    elif cfg.version.lower() == 'siamese':
        train_dataset, val_dataset = get_siamese_dataset(cfg)
        model_pl = TransferModelSiamesePL(cfg)
        train_loader = DataLoader(train_dataset, collate_fn=None, shuffle=True, num_workers=train_workers, batch_size=None)
        val_loader = DataLoader(val_dataset, collate_fn=None, num_workers=val_workers, batch_size=None, shuffle=False)

    else:
        raise ValueError('Invalid ThermoMPNN version! Must be v1, v2, or siamese')
    
    # additional params, logging, checkpoints for training
    filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = f'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath='checkpoints', filename=filename)
    logger = WandbLogger(project=cfg.project, name="test", log_model=False) if cfg.project is not None else None
    n_steps = 100 if cfg.version == 'v2' else 10
        
    trainer = pl.Trainer(callbacks=[checkpoint_callback], 
                        logger=logger, 
                        log_every_n_steps=n_steps, 
                        max_epochs=cfg.training.epochs,
                        accelerator=cfg.platform.accel, 
                        devices=1, 
                        limit_train_batches=cfg.training.batch_fraction)
    
    trainer.fit(model_pl, train_loader, val_loader)


if __name__ == "__main__":
    # config.yaml and local.yaml files are combined to assemble all runtime arguments
    if len(sys.argv) != 3:
        raise ValueError("Need to specify exactly two config files (config.yaml and local.yaml)")
    
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))
    train(cfg)
