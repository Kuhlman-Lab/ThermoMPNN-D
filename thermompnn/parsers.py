from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2


def get_v2_dataset(cfg):
    query = cfg.data.dataset.lower()
    splits = cfg.data.splits
    if query.startswith('megascale'):
        return MegaScaleDatasetv2(cfg, splits[0]), MegaScaleDatasetv2(cfg, splits[1])
    else:
        raise ValueError("Invalid training dataset '%s' selected!" % query)
