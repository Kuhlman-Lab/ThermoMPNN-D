import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from omegaconf import OmegaConf

import sys
sys.path.append('../')
from datasets import MegaScaleDataset, FireProtDataset, ddgBenchDataset
from transfer_model import get_protein_mpnn
from train_thermompnn import TransferModelPL
from protein_mpnn_utils import tied_featurize


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20, surface_threshold: int = 15, backup_atom: str = "C", chain: str = 'A') -> torch.Tensor:

    coords = xyz[basis_atom + f'_chain_{chain}']
    coords = torch.tensor(coords)
    # Compute distances and number of neighbors.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1
    # Compute centralities
    # centralities = {
    #     'all': torch.ones(num_neighbors.shape, device=num_neighbors.device),
    #     'core': num_neighbors >= core_threshold,
    #     # 'boundary': num_neighbors < core_threshold & num_neighbors > surface_threshold,
    #     'surface': num_neighbors <= surface_threshold,
    # }
    return num_neighbors


class ProteinMPNNBaseline(nn.Module):
    """Class for running ProteinMPNN as a ddG proxy predictor"""

    def __init__(self, cfg, version='v_48_020.pt'):
        super().__init__()
        self.prot_mpnn = get_protein_mpnn(cfg, version=version)

    def forward(self, pdb, mutations, tied_feat=True):
        device = next(self.parameters()).device
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(
                [pdb[0]], device, None, None, None, None, None, None, ca_only=False)

        *_, log_probs = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        out = []
        for mut in mutations:
            if mut is None:
                out.append(None)
                continue

            aa_index = ALPHABET.index(mut.mutation)
            pred = log_probs[0, mut.position, aa_index]

            out.append({
                "ddG": -torch.unsqueeze(pred, 0),
                "dTm": torch.unsqueeze(pred, 0)
            })
        return out, log_probs


def get_metrics():
    return {
        "r2": R2Score().to('cuda'),
        "mse": MeanSquaredError(squared=True).to('cuda'),
        "rmse": MeanSquaredError(squared=False).to('cuda'),
        "spearman": SpearmanCorrCoef().to('cuda'),
        "pearson":  PearsonCorrCoef().to('cuda'),
    }


def get_trained_model(model_name, config, checkpt_dir='checkpoints/', override_custom=False):
    if override_custom:
        return TransferModelPL.load_from_checkpoint(model_name, cfg=config).model
    else:
        model_loc = os.path.join(config.platform.thermompnn_dir, checkpt_dir)
        model_loc = os.path.join(model_loc, model_name)
        return TransferModelPL.load_from_checkpoint(model_loc, cfg=config).model


def run_prediction_default(name, model, dataset_name, dataset, results):
    """Standard inference for CSV/PDB based dataset"""

    max_batches = None

    metrics = {
        "ddG": get_metrics(),
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
        # print(metrics['ddG']['pearson'].compute().cpu().item(), '***')
        # quit()
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
        "ddG": get_metrics(),
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


def main(cfg, args):

    # define config for model loading
    config = {
        'data': {
            'TR': False,
            'mut_types': ['single']
        },
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'single_target': False,
            # 'mutant_embedding': False,
            # 'mlp_first': False,
            'lightattn': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)
    # from copy import deepcopy

    # cfg_ipmp_000 = deepcopy(cfg)
    # cfg_ipmp_000.model.version = 'ProteinIPMP_000_0.pt'

    # cfg_ipmp_002 = deepcopy(cfg)
    # cfg_ipmp_002.model.version = 'ProteinIPMP_002_0.pt'

    # cfg_ipmp_010 = deepcopy(cfg)
    # cfg_ipmp_010.model.version = 'ProteinIPMP_010_0.pt'

    # cfg_ipmp_020 = deepcopy(cfg)
    # cfg_ipmp_020.model.version = 'ProteinIPMP_020_0.pt'

    # cfg_ipmp_030 = deepcopy(cfg)
    # cfg_ipmp_030.model.version = 'ProteinIPMP_030_0.pt'

    # cfg_mpnn_000 = deepcopy(cfg)
    # cfg_mpnn_000.model.version = 'ProteinMPNN_000_0.pt'

    # cfg_mpnn_002 = deepcopy(cfg)
    # cfg_mpnn_002.model.version = 'ProteinMPNN_002_0.pt'

    # cfg_mpnn_010 = deepcopy(cfg)
    # cfg_mpnn_010.model.version = 'ProteinMPNN_010_0.pt'

    # cfg_mpnn_020 = deepcopy(cfg)
    # cfg_mpnn_020.model.version = 'ProteinMPNN_020_0.pt'

    # cfg_mpnn_030 = deepcopy(cfg)
    # cfg_mpnn_030.model.version = 'ProteinMPNN_030_0.pt'



    models = {
        # 'ProteinMPNN': ProteinMPNNBaseline(cfg, version='v_48_020.pt'),
        "ThermoMPNN": get_trained_model(model_name='thermoMPNN_default.pt', config=cfg, checkpt_dir='models/'),
        # "ThermoMPNN-newPDBs": get_trained_model(model_name='aug_pdb_ThermoMPNN_epoch=25_val_ddG_spearman=0.79.ckpt', config=cfg, checkpt_dir='checkpoints/'),

        # "NOSUB": get_trained_model(model_name="MLP_LA_NOSUB_2_epoch=30_val_ddG_spearman=0.79.ckpt", config=cfg, checkpt_dir='/nas/longleaf/home/dieckhau/protein-stability/enzyme-stability/checkpoints/')
        
        # "ThermoMPNN-order-invar": get_trained_model(model_name='double_mlp_init_wMUTEMB_singletarget_epoch=01_val_ddG_spearman=0.59.ckpt', 
        #                                             config=cfg),

        # "ThermoMPNN-dm": get_trained_model(model_name='double_concat_epoch=34_val_ddG_spearman=0.7.ckpt',
        #                                 config=cfg, checkpt_dir='checkpoints/'),
        # "ThermoMPNN-dm-flip": get_trained_model(model_name='double_concat_valflip_epoch=37_val_ddG_spearman=0.46.ckpt',
        #                                 config=cfg, checkpt_dir='checkpoints/'),                               
        # "MPNN_000": get_trained_model(model_name='MPNN_000_0_epoch=60_val_ddG_spearman=0.77.ckpt',
        #                                 config=cfg_mpnn_000, checkpt_dir='checkpoints/'),
        # "MPNN_002": get_trained_model(model_name='MPNN_002_0_epoch=36_val_ddG_spearman=0.78.ckpt',
        #                                 config=cfg_mpnn_002, checkpt_dir='checkpoints/'),
        # "MPNN_010": get_trained_model(model_name='MPNN_010_0_epoch=20_val_ddG_spearman=0.79.ckpt',
        #                     config=cfg_mpnn_010, checkpt_dir='checkpoints/'),
        # "MPNN_020": get_trained_model(model_name='MPNN_020_0_epoch=27_val_ddG_spearman=0.8.ckpt',
        #                     config=cfg_mpnn_020, checkpt_dir='checkpoints/'),
        # "MPNN_030": get_trained_model(model_name='MPNN_030_0_epoch=46_val_ddG_spearman=0.79.ckpt',
        #                     config=cfg_mpnn_030, checkpt_dir='checkpoints/'),       
        # "IPMP_000": get_trained_model(model_name='IPMP_000_0_epoch=31_val_ddG_spearman=0.78.ckpt',
        #                                 config=cfg_ipmp_000, checkpt_dir='checkpoints/'),
        # "IPMP_002": get_trained_model(model_name='IPMP_002_0_epoch=58_val_ddG_spearman=0.78.ckpt',
        #                                 config=cfg_ipmp_002, checkpt_dir='checkpoints/'),
        # "IPMP_010": get_trained_model(model_name='IPMP_010_0_epoch=33_val_ddG_spearman=0.78.ckpt',
        #                     config=cfg_ipmp_010, checkpt_dir='checkpoints/'),
        # "IPMP_020": get_trained_model(model_name='IPMP_020_0_epoch=40_val_ddG_spearman=0.79.ckpt',
        #                     config=cfg_ipmp_020, checkpt_dir='checkpoints/'),
        # "IPMP_030": get_trained_model(model_name='IPMP_030_0_epoch=50_val_ddG_spearman=0.79.ckpt',
        #                     config=cfg_ipmp_030, checkpt_dir='checkpoints/'),  

        # "ThermoMPNN_CV0": get_trained_model(model_name='thermoMPNN_CV0.pt', config=cfg, checkpt_dir='models/'),
        # "ThermoMPNN_CV1": get_trained_model(model_name='thermoMPNN_CV1.pt', config=cfg, checkpt_dir='models/'),
        # "ThermoMPNN_CV2": get_trained_model(model_name='thermoMPNN_CV2.pt', config=cfg, checkpt_dir='models/'),
        # "ThermoMPNN_CV3": get_trained_model(model_name='thermoMPNN_CV3.pt', config=cfg, checkpt_dir='models/'),
        # "ThermoMPNN_CV4": get_trained_model(model_name='thermoMPNN_CV4.pt', config=cfg, checkpt_dir='models/'),

    }
    from datasets import MegaScaleDatasetExperimental

    # to load megascale csvs from alternate location
    # cfg.data_loc.megascale_pdbs = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/structure_studies/megascale-all/af2/'
    cfg.data_loc.fireprot_pdbs = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/structure_studies/fireprot-HF/af2/'

    misc_data_loc = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data'
    
    datasets = {}
    pdb = '1QND'
    for n in range(1, 21):
        datasets[pdb + '_' + str(n)] = FireProtDataset(cfg, 'homologue-free', model_no=n, pdb_current=pdb)
    
    # datasets = {
        # "Megascale-train": MegaScaleDataset(cfg, "train"),
        # "Megascale-val": MegaScaleDataset(cfg, "val"),
        # "Megascale-test": MegaScaleDataset(cfg, "test"),
        
        # testing of standard (AF2 & Rosetta mixture), AF2 only, and Experimental structure performance
        # "Megascale-test-CV0": MegaScaleDataset(cfg, "cv_test_0"),
        # "Megascale-test-af2-CV0": MegaScaleDataset(cfg, "cv_test_0"),
        # "Megascale-test-exp-CV0": MegaScaleDatasetExperimental(cfg, "cv_test_0"),
        
        # "Megascale-test-CV1": MegaScaleDataset(cfg, "cv_test_1"),
        # "Megascale-test-af2-CV1": MegaScaleDataset(cfg, "cv_test_1"),
        # "Megascale-test-exp-CV1": MegaScaleDatasetExperimental(cfg, "cv_test_1"),
        
        # "Megascale-test-CV2": MegaScaleDataset(cfg, "cv_test_2"),
        # "Megascale-test-af2-CV2": MegaScaleDataset(cfg, "cv_test_2"),
        # "Megascale-test-exp-CV2": MegaScaleDatasetExperimental(cfg, "cv_test_2"),
        
        # "Megascale-test-CV3": MegaScaleDataset(cfg, "cv_test_3"),
        # "Megascale-test-af2-CV3": MegaScaleDataset(cfg, "cv_test_3"),
        # "Megascale-test-exp-CV3": MegaScaleDatasetExperimental(cfg, "cv_test_3"),
        
        # "Megascale-test-CV4": MegaScaleDataset(cfg, "cv_test_4"),
        # "Megascale-test-af2-CV4": MegaScaleDataset(cfg, "cv_test_4"),
        # "Megascale-test-exp-CV4": MegaScaleDatasetExperimental(cfg, "cv_test_4"),
        
        # "Fireprot-test": FireProtDataset(cfg, "test"),

        # "Fireprot-homologue-free": FireProtDataset(cfg, "homologue-free", model_no=1, pdb_current='2ABD'),


        # "P53": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv')),
        
        # "P53-rev": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv'), flip=True),
        
        # "MYOGLOBIN": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv')),
        # "MYOGLOBIN-rev": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv'), flip=True),

        # "SSYM_dir": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')),
        # "SSYM_inv": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')),
        # "S669": ddgBenchDataset(cfg, pdb_dir=os.path.join(misc_data_loc, 'S669/pdbs'),
        #                        csv_fname=os.path.join(misc_data_loc, 'S669/s669_clean_dir.csv')),
    # }

    # quit()

    results = []

    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            if args.keep_preds:
                print('Keeping preds!')
                results = run_prediction_keep_preds(name, model, dataset_name, dataset, results, centrality=args.centrality)
            else:
                results = run_prediction_default(name, model, dataset_name, dataset, results)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("ThermoMPNN_metrics.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--keep_preds', action='store_true', default=False, help='Save raw model predictions as csv')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). '
                             'Only used if --keep_preds is enabled.')

    args = parser.parse_args()
    cfg = OmegaConf.load("../local.yaml")
    with torch.no_grad():
        main(cfg, args)
