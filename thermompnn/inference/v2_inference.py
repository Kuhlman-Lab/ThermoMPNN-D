import torch
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, FireProtDatasetv2, ddgBenchDatasetv2, tied_featurize_mut, BinderSSMDataset, BinderSSMDatasetOmar, SKEMPIDataset, S487Dataset, PPIDataset
from thermompnn.inference.inference_utils import get_metrics_full
from thermompnn.model.v2_model import batched_index_select


def run_prediction_batched(name, model, dataset_name, dataset, results, keep=True, zero_shot=False, cfg=None):
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
    if 'megascale' not in dataset_name:
        batch_size = 1  # larger batches will fail due to different chain IDs - fix later
    
    # centrality analysis (SKEMPI)
    # muts = (dataset.df.WT_AA + dataset.df.MUT_CHAIN + dataset.df.MUT_POS.astype(str) + dataset.df.MUT_AA).values
    # wtns = (dataset.df.PDB_ID + '_' + dataset.df.CHAINS1 + '_' + dataset.df.CHAINS2).values
    # cent = []
    # ref = pd.read_csv('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/SKEMPIv2/SKEMPI_v2_database.csv',sep=';')
    # for mm, wtn in tqdm(zip(muts, wtns)):
    #     sel = ref.loc[(ref['#Pdb'] == wtn) & (ref['Mutation(s)_cleaned'] == mm), 'Mutation_Location(s)'].values[0]
    #     cent.append(sel)
    
    # centrality analysis (binderSSM)
    # cent = {}
    # for p in dataset.pdb_data.keys():
    #     # grab cross-chain neighbor count for each p
    #     # impute for every mutation
    #     data = dataset.pdb_data[p]
    #     binder_xyz = np.stack(data['coords_chain_A']['CA_chain_A']) # [Lb, 3]
    #     target_xyz = np.stack(data['coords_chain_B']['CA_chain_B']) # [Lt, 3]
    #     from scipy.spatial.distance import cdist
    #     d = cdist(binder_xyz, target_xyz) # [Lb, Lt]
    #     neigh = np.sum(d < 10, axis=-1) # [Lb]
    #     cent[p] = neigh

    # pdb = dataset.df.ssm_parent.values
    # pos = dataset.df.pos.values
    
    # cent_l = []
    # for pdi, po in tqdm(zip(pdb, pos)):
    #     c = cent[pdi][int(po) - 1]
    #     cent_l.append(c)

    loader = DataLoader(dataset, collate_fn=lambda b: tied_featurize_mut(b, side_chains=cfg.data.get('side_chains', False), esm=cfg.model.get('auxiliary_embedding', '') == 'localESM'), 
                        shuffle=False, num_workers=cfg.training.get('num_workers', 8), batch_size=cfg.training.get('batch_size', 256))

    batches = []
    for i, batch in enumerate(tqdm(loader)):

        if batch is None:
            continue
        if cfg.model.get('auxiliary_embedding', '') == 'localESM':
            X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb = batch
            esm_emb = esm_emb.to(device)
        else:
            X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch

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
        atom_mask = torch.Tensor(atom_mask).to(device)

        if cfg.model.get('auxiliary_embedding', '') == 'localESM':
            pred, _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb)
        elif cfg.model.get('aggregation', '') == 'siamese':
            # average both siamese network passes
            predA, predB = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            # print(torch.abs(predA[0] - predB[0]))
            pred = torch.mean(torch.cat([predA, predB], dim=-1), dim=-1)
        elif not zero_shot:
            pred, _ = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)
            # pred = torch.argmax(pred, dim=-1).unsqueeze(-1).to(torch.float32) # for classifer inference compatibility
        else:
            pred = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)[-2]
            pred = zero_shot_convert(pred, mut_positions, mut_mutant_AAs, mut_wildtype_AAs)
            # pred = zero_shot_convert(pred, mut_positions, mut_mutant_AAs) # absolute logits are less predictive than relative logits
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(-1)

        for metric in metrics["ddG"].values():
            metric.update(torch.squeeze(pred, dim=-1), torch.squeeze(mut_ddGs, dim=-1))

        if max_batches is not None and i >= max_batches:
            break
        
        preds += list(torch.squeeze(pred, dim=-1).detach().cpu())
        ddgs += list(torch.squeeze(mut_ddGs, dim=-1).detach().cpu())
        batches += [i for p in range(len(pred))]   
    
    print('%s mutations evaluated' % (str(len(ddgs))))
    
    if keep:
        preds, ddgs = np.squeeze(preds), np.squeeze(ddgs)

        if 'megascale' in dataset_name:
            tmp = pd.DataFrame({'ddG_pred': preds, 
            'ddG_true': ddgs, 
            'batch': batches, 
            'mut_type': dataset.df.mut_type, 
            'WT_name': dataset.df.WT_name})

        elif 'binder' in dataset_name:
            tmp = pd.DataFrame({'ddG_pred': preds, 
                                'ddG_true': ddgs, 
                                'batch': batches,
                                'region': dataset.df.region,
                                # 'mut_type': dataset.df.pos + dataset.df.mut_to,
                                # 'mut_type': dataset.df.wtAA + dataset.df.pos + dataset.df.mutAA, 
                                'WT_name': dataset.df.ssm_parent, 
                                })
        elif 'fireprot' in dataset_name:
            dataset.df.pdb_position = dataset.df.pdb_position.astype(str)
            tmp = pd.DataFrame({'ddG_pred': preds, 
                                'ddG_true': ddgs, 
                                'batches': batches, 
                                'mut_type': dataset.df.wild_type + dataset.df.pdb_position + dataset.df.mutation,
                                'WT_name': dataset.df.pdb_id_corrected})
        elif 'SKEMPI' in dataset_name:
            tmp = pd.DataFrame({'ddG_pred': preds, 
                                'ddG_true': ddgs, 
                                'batch': batches, 
                                'wtAA': dataset.df.WT_AA, 
                                'mutAA': dataset.df.MUT_AA, 
                                'POS': dataset.df.MUT_POS, 
                                'WT_name': dataset.df.PDB_ID})
        elif 's487' in dataset_name:
            tmp = pd.DataFrame({'ddG_pred': preds, 
                                'ddG_true': ddgs, 
                                'batch': batches, 
                                'wtAA': dataset.df.WT_AA, 
                                'mutAA': dataset.df.MUT_AA, 
                                'POS': dataset.df.POS, 
                                'WT_name': dataset.df.PDB})
        else:
            # if 'ptmul' in dataset_name: # manually correct for subset inference df size mismatch
                # dataset.df = dataset.df.loc[dataset.df.NMUT > 9].reset_index(drop=True)
            tmp = pd.DataFrame({'ddG_pred': preds, 
            'ddG_true': ddgs, 
            'batch': batches, 
            'mut_type': dataset.df.MUTS, 
            'WT_name': dataset.df.PDB})
        
        # tmp['CENTRALITY'] = cent
        # tmp['CENTRALITY'] = cent_l
        # tmp = tmp.reset_index(drop=True)

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
        'fireprot': FireProtDatasetv2, 
        'binder-SSM': BinderSSMDataset,
        'ddgbench': ddgBenchDatasetv2
    }
    dataset = cfg.data.dataset
    split = cfg.data.splits[0]
    # common splits: test, test_cdna2, homologue-free
    print(dataset, )
    if dataset == 'megascale' and split == 'test_cdna2':
        cfg.data_loc.megascale_csv = '/home/hdieckhaus/scripts/ThermoMPNN/data/cdna_mutate_everything/cdna2_test_ThermoMPNN.csv'

    if dataset == 'megascale' or dataset == 'fireprot':
        ds = ds_all[dataset]
        return ds(cfg, split)

    elif dataset == 'binderSSM-omar':
        print('Loading Omar SSM Dataset')
        csv_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/omar-processed-data/ssm.sc')
        pdb_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/omar-processed-data/parents')
        split_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/omar-processed-data/splits/ssm_splits.pkl')
        return BinderSSMDatasetOmar(cfg, split, csv_loc, pdb_loc, split_loc)

    elif dataset == 'binder-SSM':
        
        ds = ds_all[dataset]
        csv_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/Binder-SSM-Dataset.csv')
        pdb_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/parents')
        split_loc = os.path.join(cfg.data_loc.misc_data, 'binder-SSM/ssm_split_henry.pkl')
        return ds(cfg, split, 
                  csv_file=csv_loc, 
                  pdb_loc=pdb_loc, 
                  split_file=split_loc)

    elif dataset == 'SKEMPI':
        csvf = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/SKEMPIv2/SKEMPI_v2_single.csv'
        pdbd = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/SKEMPIv2/PDBs'
        dataset = SKEMPIDataset(cfg, csv_file=csvf, pdb_loc=pdbd, split=split)
        return dataset

    elif dataset == 's487':
        csvf = os.path.join(cfg.data_loc.misc_data, 'SKEMPIv2/s487.csv')
        pdbd = os.path.join(cfg.data_loc.misc_data, 'SKEMPIv2/PDBs')
        return S487Dataset(cfg, csv_file=csvf, pdb_loc=pdbd)

    elif dataset == 'mdm2-p53':
        csvf = os.path.join(cfg.data_loc.misc_data, 'case_studies/MDM2-p53/mdm2_p53.tsv')
        pdbd = os.path.join(cfg.data_loc.misc_data, 'case_studies/MDM2-p53')
        return PPIDataset(cfg, csv_file=csvf, pdb_loc=pdbd)

    else:
        ds = ds_all['ddgbench']
        flip = False

        if dataset == 's669':
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'S669/pdbs')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'S669/s669_clean_dir.csv')

        elif dataset == 'ssym':
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/SSYM/pdbs')
            if split == 'dir':
                csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')
            else:
                csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')

        elif dataset == 'p53':
            if split != 'dir':  # handle inverse mutations (w/Rosetta structures)
                flip = True
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/P53/pdbs')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/P53/p53_clean.csv')

        elif dataset == 'myoglobin':
            if split != 'dir':
                flip = True
            pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/MYOGLOBIN/pdbs')
            csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')

        elif dataset == 'ptmul':
            if split != 'dir': # ptmul mutateeverything splits
                print('loading ptmul with alternate splits/curation from MutateEverything paper')
                pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/pdbs')
                csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/ptmul-5fold-mutateeverything_FINAL.csv')
            else:    
                pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/pdbs')
                csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/ptmul-5fold.csv')

        return ds(cfg, pdb_loc, csv_loc, flip=flip)


def zero_shot_convert(preds, positions, mut_AAs, wt_AAs=None):
    """Convert raw ProteinMPNN log-probs into ddG pseudo-values"""
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    # index positions 
    preds = batched_index_select(preds, 1, positions)
    # index mutAA indices
    mut_logs = batched_index_select(preds, 2, mut_AAs)
    mut_logs = torch.squeeze(torch.squeeze(mut_logs, -1), -1)
    if wt_AAs is not None:
        wt_logs = torch.squeeze(torch.squeeze(batched_index_select(preds, 2, wt_AAs), -1), -1)
        mut_logs = wt_logs - mut_logs
    else:
        mut_logs = -1 * mut_logs

    return mut_logs
