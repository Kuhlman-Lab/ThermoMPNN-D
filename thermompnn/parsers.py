import os 

from thermompnn.datasets.v1_datasets import MegaScaleDataset, FireProtDataset
from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, BinderSSMDataset, BinderSSMDatasetOmar, SKEMPIDataset
from thermompnn.datasets.siamese_datasets import MegaScaleDatasetSiamese, MegaScaleDatasetSiamesePt


def get_v1_dataset(cfg):
    valid_ds = ['megascale', 'fireprot', 'combo']
    query = cfg.dataset.lower()
    if not query.startswith(valid_ds):
        raise ValueError("Invalid v1 dataset selected!")
    
    datasets = {
        'megascale': MegaScaleDataset, 
        'fireprot': FireProtDataset, 
    }
    
    if query in datasets.keys():
        train, val = 'train', 'val'
        ds = datasets[cfg.dataset]
    else:
        if query == 'megascale_s669':
            train, val = 'train_s669', 'val'
        elif query.startswith('megascale_cv'): 
            train, val = f'cv_train_{cfg.dataset[-1]}', f'cv_val_{cfg.dataset[-1]}'
            ds = datasets[cfg.dataset]
        else:
            raise ValueError("Invalid dataset specified!")
    return ds(cfg, train), ds(cfg, val)


def get_v2_dataset(cfg):
    query = cfg.data.dataset.lower()
    splits = cfg.data.splits
    if query.startswith('megascale'):
        return MegaScaleDatasetv2(cfg, splits[0]), MegaScaleDatasetv2(cfg, splits[1])
    elif query.startswith('binder'):
        if query.endswith('omar'):
            print('Loading Omar SSM Dataset')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/omar-processed-data/ssm.sc')
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/omar-processed-data/parents')
            split_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/omar-processed-data/splits/ssm_splits.pkl')
            return BinderSSMDatasetOmar(cfg, splits[0], csv_loc, pdb_loc, split_loc), BinderSSMDatasetOmar(cfg, splits[1], csv_loc, pdb_loc, split_loc)
        else:
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/Binder-SSM-Dataset.csv')
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/parents')
            split_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/ssm_split_henry.pkl')
            return BinderSSMDataset(cfg, splits[0], csv_loc, pdb_loc, split_loc), BinderSSMDataset(cfg, splits[1], csv_loc, pdb_loc, split_loc)
    elif query.startswith('skempi'):
        print('Loading SKEMPI dataset!')
        csv_loc = os.path.join(cfg.data_loc.misc_data, 'SKEMPIv2/SKEMPI_v2_single.csv')
        pdb_loc = os.path.join(cfg.data_loc.misc_data, 'SKEMPIv2/PDBs')
        return SKEMPIDataset(cfg, csv_loc, pdb_loc, splits[0]), SKEMPIDataset(cfg, csv_loc, pdb_loc, splits[1])
    else:
        raise ValueError("Invalid training dataset '%s' selected!" % query)

def get_siamese_dataset(cfg):
    # valid_ds = ['megascale', 'megascale_pt']
    query = cfg.dataset.lower()
    assert query.contains('megascale')
    if query.endswith('pt'): # contains mutant structures
        return MegaScaleDatasetSiamesePt(cfg, 'train'), MegaScaleDatasetSiamesePt(cfg, 'val')   
    else: # lazy data augmentation
        return MegaScaleDatasetSiamese(cfg, 'train'), MegaScaleDatasetSiamese(cfg, 'val')