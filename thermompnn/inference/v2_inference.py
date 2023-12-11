import torch

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, FireProtDatasetv2, ddgBenchDatasetv2
from thermompnn.inference.inference_utils import get_metrics_full


def run_prediction_batched(name, model, dataset_name, dataset, results, keep=True):
    """Standard inference for CSV/PDB based dataset in batched models"""

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    max_batches = None
    metrics = {
        "ddG": get_metrics_full(),
    }
    for m in metrics['ddG'].values():
        m = m.to(device)
    
    print('Testing Model %s on dataset %s' % (name, dataset_name))
    preds, ddgs = [], []
    for i, batch in enumerate(tqdm(dataset)):
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = batch
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

        pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs)

        for metric in metrics["ddG"].values():
            metric.update(torch.squeeze(pred, dim=-1), torch.squeeze(mut_ddGs, dim=-1))

        if max_batches is not None and i >= max_batches:
            break
        
        preds += list(torch.squeeze(pred, dim=-1).detach().cpu())
        ddgs += list(torch.squeeze(mut_ddGs, dim=-1).detach().cpu())

    preds, ddgs = np.squeeze(preds), np.squeeze(ddgs)
    tmp = pd.DataFrame({'ddG_pred': preds, 'ddG_true': ddgs})
    print(tmp.head)
    tmp.to_csv('ThermoMPNN_raw_preds_batched.csv')

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


def load_v2_dataset(cfg):
    """Parses input config and sets up proper dataset for INFERENCE only"""

    ds_all = {
        'megascale': MegaScaleDatasetv2,  
        'fireprot': FireProtDatasetv2, 
        'ddgbench': ddgBenchDatasetv2
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
