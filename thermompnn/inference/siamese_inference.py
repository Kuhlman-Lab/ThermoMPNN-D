import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import pandas as pd

from thermompnn.inference.inference_utils import get_metrics_full
from thermompnn.datasets.siamese_datasets import SsymDatasetSiamese, ddgBenchDatasetSiamese, FireProtDatasetSiamese, MegaScaleDatasetSiamese


def load_siamese_dataset(cfg):
    """Parses input config and sets up proper dataset for INFERENCE only"""

    ds_all = {
        'megascale': MegaScaleDatasetSiamese,  
        'fireprot': FireProtDatasetSiamese, 
        'ddgbench': ddgBenchDatasetSiamese, 
        'ssym': SsymDatasetSiamese
    }
    # dataset format: dataset-split
    parts = cfg.dataset.split('-')
    prefix = parts[0]
    split = parts[-1] if len(parts) > 1 else 'dir'

    if prefix == 'megascale' or prefix == 'fireprot':
        ds = ds_all[prefix]
        return ds(cfg, split)
    else:
        ds = ds_all['ddgbench']
        flip = False

        if prefix == 's669':
            pdb_loc = os.path.join(cfg.misc_data, 'S669/pdbs')
            csv_loc = os.path.join(cfg.misc_data, 'S669/s669_clean_dir.csv')

        elif prefix == 'ssym':
            pdb_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/SSYM/pdbs')
            if prefix == 'dir':
                csv_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')
            else:
                csv_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')

        elif prefix == 'p53':
            if split != 'dir':  # handle inverse mutations (w/Rosetta structures)
                flip = True
            pdb_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/P53/pdbs')
            csv_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/P53/p53_clean.csv')

        elif prefix == 'myoglobin':
            if split != 'dir':
                flip = True
            pdb_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/MYOGLOBIN/pdbs')
            csv_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')

        return ds(cfg, pdb_loc, csv_loc, flip=flip)


def run_prediction_siamese(name, model, dataset_name, dataset, results, keep=False, use_both=True):
    """Standard inference for CSV/PDB based dataset in batched models"""

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    max_batches = None
    metrics = {
        "ddG": get_metrics_full(),
    }
    for m in metrics['ddG'].values():
        m = m.to(device)
    print('Use both?', use_both)
    print('Testing Model %s on dataset %s' % (name, dataset_name))
    preds, ddgs = [], []
    loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=7, batch_size=None)
    for i, batch in enumerate(tqdm(loader)):
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = batch[0]
        X = X.to(device)
        S = S.to(device)
        mask = mask.to(device)
        lengths = torch.Tensor(lengths).to(device)
        chain_M = chain_M.to(device)
        chain_encoding_all = chain_encoding_all.to(device)
        residue_idx = residue_idx.to(device)
        mut_positions = mut_positions.to(device)
        mut_wildtype_AAs = mut_wildtype_AAs.to(device)
        mut_mutant_AAs = mut_mutant_AAs.to(device)
        mut_ddGs = mut_ddGs.to(device)
        new_batch_0 = (X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs)

        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = batch[1]
        X = X.to(device)
        S = S.to(device)
        mask = mask.to(device)
        lengths = torch.Tensor(lengths).to(device)
        chain_M = chain_M.to(device)
        chain_encoding_all = chain_encoding_all.to(device)
        residue_idx = residue_idx.to(device)
        mut_positions = mut_positions.to(device)
        mut_wildtype_AAs = mut_wildtype_AAs.to(device)
        mut_mutant_AAs = mut_mutant_AAs.to(device)
        mut_ddGs = mut_ddGs.to(device)
        new_batch_1 = [X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs]

        # pred_fwd, pred_inv = model(new_batch_1, new_batch_0)  # to make lazy reverse mutant dataset
        pred_fwd, pred_inv = model(new_batch_0, new_batch_1)
        if use_both:
            target_ddg = (pred_fwd - pred_inv) / 2  # for true siamese, take avg of both dirs
        else:
            target_ddg = pred_fwd
        sigma = (pred_fwd + pred_inv) / 2
        for metric in metrics["ddG"].values():
            metric.update(torch.squeeze(target_ddg, dim=-1), torch.squeeze(mut_ddGs * -1, dim=-1))

        if max_batches is not None and i >= max_batches:
            break
        
        preds += list(torch.squeeze(target_ddg, dim=-1).detach().cpu())
        ddgs += list(torch.squeeze(mut_ddGs, dim=-1).detach().cpu())

    preds, ddgs = np.squeeze(preds), np.squeeze(ddgs)
    if keep:
        tmp = pd.DataFrame({'ddG_pred': preds, 'ddG_true': ddgs})
        print(tmp.head)
        tmp.to_csv(f'{name}_{dataset_name}_preds_raw_siamese.csv')
    
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                print(met_name, column[f"{dtype} {met_name}"])
            except ValueError:
                pass
    results.append(column)
    return results

