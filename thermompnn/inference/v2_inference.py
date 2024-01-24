import torch
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, FireProtDatasetv2, ddgBenchDatasetv2, MegaScaleDatasetv2Pt, FireProtDatasetv2Confidence
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
    
    model = model.eval()
    model = model.cuda()
    
    print('Testing Model %s on dataset %s' % (name, dataset_name))
    preds, ddgs = [], []
    
    loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=8, batch_size=None)
    # loader = dataset
    batches = []
    for i, batch in enumerate(tqdm(loader)):
        # for conf model
        if batch is None:
            # print('Skipping batch %s for lack of mutation' % str(i))
            # preds += [-10000 for n in ]
            continue
        # X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, ddG_err = batch
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
        
        # ddG_err = ddG_err.to(device)

        pred, _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs)

        # for conf model validation
        # mut_ddGs = torch.abs(pred - mut_ddGs)
        # pred = _

        for metric in metrics["ddG"].values():
            metric.update(torch.squeeze(pred, dim=-1), torch.squeeze(mut_ddGs, dim=-1))

        if max_batches is not None and i >= max_batches:
            break
        
        preds += list(torch.squeeze(pred, dim=-1).detach().cpu())
        ddgs += list(torch.squeeze(mut_ddGs, dim=-1).detach().cpu())
        batches += [i for p in range(len(pred))]
    
        # quit()
        
        # TODO save batches as pt of values for independent confidence model
        # diff = torch.abs(torch.squeeze(pred, dim=-1) - torch.squeeze(mut_ddGs, dim=-1))

        # results = [X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, diff]
        # fpath = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/conf_batches/%s' % 'SSYM-inv', f'batch_{i}.pt')
        # fpath = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/conf_batches/%s' % 'fireprot-HF', f'batch_{i}.pt')
        # torch.save(results, fpath)

    print('%s mutations evaluated' % (str(len(ddgs))))
    
    if keep:
        preds, ddgs = np.squeeze(preds), np.squeeze(ddgs)
        tmp = pd.DataFrame({'ddG_pred': preds, 'ddG_true': ddgs, 'batch': batches})
        print(tmp.head)

        tmp.to_csv(f'ThermoMPNN_{os.path.basename(name).removesuffix(".ckpt")}_{dataset_name}_preds.csv')

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
        # 'megascale': MegaScaleDatasetv2Pt,
        'fireprot': FireProtDatasetv2, 
        # 'fireprot': FireProtDatasetv2Confidence,
        'ddgbench': ddgBenchDatasetv2
    }
    # dataset format: dataset-split
    parts = cfg.dataset.split('-')
    prefix = parts[0]
    split = '-'.join(parts[1:]) if len(parts) > 1 else 'dir'

    if prefix == 'megascale' or prefix == 'fireprot':
        ds = ds_all[prefix]
        return ds(cfg, split)
    else:
        ds = ds_all['ddgbench']
        flip = False

        if prefix == 's669':
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'S669/pdbs')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'S669/s669_clean_dir.csv')

        elif prefix == 'ssym':
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/SSYM/pdbs')
            if split == 'dir':
                csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')
            else:
                csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')

        elif prefix == 'p53':
            if split != 'dir':  # handle inverse mutations (w/Rosetta structures)
                flip = True
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/P53/pdbs')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/P53/p53_clean.csv')

        elif prefix == 'myoglobin':
            if split != 'dir':
                flip = True
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/MYOGLOBIN/pdbs')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')

        elif prefix == 'ptmul':
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/pdbs')
            # csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/ptmul-5fold-singles.csv')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/ptmul-5fold.csv')

        return ds(cfg, pdb_loc, csv_loc, flip=flip)
