
from thermompnn.datasets.v1_datasets import MegaScaleDataset, FireProtDataset, ComboDataset
from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2
from thermompnn.datasets.siamese_datasets import MegaScaleDatasetSiamese, MegaScaleDatasetSiamesePt


def get_v1_dataset(cfg):
    valid_ds = ['megascale', 'fireprot', 'combo']
    query = cfg.dataset.lower()
    if not query.startswith(valid_ds):
        raise ValueError("Invalid v1 dataset selected!")
    
    datasets = {
        'megascale': MegaScaleDataset, 
        'fireprot': FireProtDataset, 
        'combo': ComboDataset
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
    assert query.startswith('megascale')
    return MegaScaleDatasetv2(cfg, splits[0]), MegaScaleDatasetv2(cfg, splits[1])


def get_siamese_dataset(cfg):
    # valid_ds = ['megascale', 'megascale_pt']
    query = cfg.dataset.lower()
    assert query.contains('megascale')
    if query.endswith('pt'): # contains mutant structures
        return MegaScaleDatasetSiamesePt(cfg, 'train'), MegaScaleDatasetSiamesePt(cfg, 'val')   
    else: # lazy data augmentation
        return MegaScaleDatasetSiamese(cfg, 'train'), MegaScaleDatasetSiamese(cfg, 'val')