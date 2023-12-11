import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import pandas as pd

from thermompnn.inference.inference_utils import get_metrics_full

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

