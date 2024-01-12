
from thermompnn.datasets.v1_datasets import MegaScaleDataset, FireProtDataset, ComboDataset
from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, MegaScaleDatasetv2Pt, MegaScaleDatasetv2Aug, MegaScaleDatasetv2Rebatched
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
    # valid_ds = ['megascale', 'megascale_pt', 'megascale-de0']
    query = cfg.dataset.lower()
    assert query.startswith('megascale')
    if query.endswith('pt'): # contains mutant structures
        return MegaScaleDatasetv2Pt(cfg, 'train'), MegaScaleDatasetv2Pt(cfg, 'val')
    elif query.endswith('rebatched'): # contains mutant structures
        print('loading rebatched dataset with ptmul-filtered training dataset')
        return MegaScaleDatasetv2Rebatched(cfg, 'train_ptmul'), MegaScaleDatasetv2Rebatched(cfg, 'val')
    else: # un-augmented or lazy data augmentation
        if query.startswith('megascale-de'): # special case here - deep ensemble training
            train, val = f'de_train_{query[-1]}', f'de_val_{query[-1]}'
            # grab megascale dataset object
            ds = MegaScaleDatasetv2
            return ds(cfg, train), ds(cfg, val)

        return MegaScaleDatasetv2(cfg, 'train'), MegaScaleDatasetv2(cfg, 'val')


def get_siamese_dataset(cfg):
    # valid_ds = ['megascale', 'megascale_pt']
    query = cfg.dataset.lower()
    assert query.contains('megascale')
    if query.endswith('pt'): # contains mutant structures
        return MegaScaleDatasetSiamesePt(cfg, 'train'), MegaScaleDatasetSiamesePt(cfg, 'val')   
    else: # lazy data augmentation
        return MegaScaleDatasetSiamese(cfg, 'train'), MegaScaleDatasetSiamese(cfg, 'val')