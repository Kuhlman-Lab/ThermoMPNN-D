import os
import pandas as pd
from tqdm import tqdm

from thermompnn.datasets.v1_datasets import MegaScaleDataset, FireProtDataset, ddgBenchDataset
from thermompnn.inference.inference_utils import compute_centrality, get_metrics_full


def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None

    metrics = {
        "ddG": get_metrics_full(),
    }
    print('Testing Model %s on dataset %s' % (name, dataset_name))
    skipped = 0
    for i, batch in enumerate(tqdm(dataset)):
        pdb, mutations = batch
        if pdb is None:  # skip missing pdbs
            print('Skipping batch due to missing PDB')
            skipped += 1
            continue
        
        pred, _ = model(pdb, mutations)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

        if max_batches is not None and i >= max_batches:
            break

    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
                print(column[f"{dtype} {met_name}"])
            except ValueError:
                pass
    results.append(column)
    if skipped != 0:
        print('SKIPPED %s batches due to missing PDBs' % (skipped))
    return results


def run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=False):
    """Inference for CSV/PDB based dataset saving raw predictions for later analysis."""
    row = 0
    max_batches = None
    raw_pred_df = pd.DataFrame(
        columns=['WT Seq', 'Model', 'Dataset', 'ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation',
                 'neighbors', 'best_AA'])
    metrics = {
        "ddG": get_metrics_full(),
    }
    skipped = 0
    print('Running model %s on dataset %s' % (name, dataset_name))
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=8, batch_size=None)
    for i, batch in enumerate(tqdm(dataset)):
        mut_pdb, mutations = batch
        # print(mut_pdb, mutations)
        if mut_pdb is None:  # skip missing pdbs
            print('Skipping batch due to missing PDB')
            skipped += 1
            continue
        
        pred, _ = model(mut_pdb, mutations)

        if centrality:
            coord_chain = [c for c in mut_pdb[0].keys() if 'coords' in c][0]
            chain = coord_chain[-1]
            neighbors = compute_centrality(mut_pdb[0][coord_chain], basis_atom='CA', backup_atom='C', chain=chain,
                                           radius=10.)

        for mut, out in zip(mutations, pred):
            if mut.ddG is not None:
                mut.ddG = mut.ddG.to('cuda')
                for metric in metrics["ddG"].values():
                    metric.update(out["ddG"], mut.ddG)

                # assign raw preds and useful details to df
                col_list = ['ddG_true', 'ddG_pred', 'position', 'wildtype', 'mutation', 'pdb']
                val_list = [mut.ddG.cpu().item(), out["ddG"].cpu().item(), mut.position, mut.wildtype,
                            mut.mutation, mut.pdb.strip('.pdb')]
                for col, val in zip(col_list, val_list):
                    raw_pred_df.loc[row, col] = val

                if centrality:
                    raw_pred_df.loc[row, 'neighbors'] = neighbors[mut.position].cpu().item()

            raw_pred_df.loc[row, 'Model'] = name
            raw_pred_df.loc[row, 'Dataset'] = dataset_name
            if 'Megascale' not in dataset_name: # different pdb column formatting
                key = mut.pdb
            else:
                key = mut.pdb + '.pdb'
            if 'S669' not in dataset_name: # S669 is missing WT seq info - omit to prevent error
                raw_pred_df.loc[row, 'WT Seq'] = dataset.wt_seqs[key]
            row += 1

        if max_batches is not None and i >= max_batches:
            break
        # break  # TODO remove
    column = {
        "Model": name,
        "Dataset": dataset_name,
    }
    for dtype in ["ddG"]:  # , "dTm"]:
        for met_name, metric in metrics[dtype].items():
            try:
                column[f"{dtype} {met_name}"] = metric.compute().cpu().item()
            except ValueError:
                pass
    results.append(column)
    
    if skipped != 0:
        print('Skipped %s batches due to missing PDBs' % (skipped))
    raw_pred_df.to_csv(name + '_' + dataset_name + "_raw_preds.csv")
    del raw_pred_df

    return results


def load_v1_dataset(cfg):
    """Parses input config and sets up proper dataset for INFERENCE only"""

    ds_all = {
        'megascale': MegaScaleDataset,  
        'fireprot': FireProtDataset, 
        'ddgbench': ddgBenchDataset
    }
    # dataset format: dataset-split
    parts = cfg.dataset.split('-')
    prefix = parts[0]
    split = parts[-1] if len(parts) > 1 else None

    if prefix == 'megascale' or prefix == 'fireprot':
        ds = ds_all[prefix]
        return ds(cfg, split)
    else:
        ds = ds_all['ddgbench']

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
            pdb_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/P53/pdbs')
            csv_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/P53/p53_clean.csv')

        elif prefix == 'myoglobin':
            pdb_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/MYOGLOBIN/pdbs')
            csv_loc = os.path.join(cfg.misc_data, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')

        return ds(cfg, pdb_loc, csv_loc)
