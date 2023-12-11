import torch
from torch import nn
from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef, PearsonCorrCoef

import argparse
import omegaconf as OmegaConf
import pandas as pd

from thermompnn.model.modules import get_protein_mpnn
from thermompnn.protein_mpnn_utils import tied_featurize
from thermompnn.inference.v1_inference import load_v1_dataset, run_prediction_default, run_prediction_keep_preds
from thermompnn.inference.v2_inference import load_v2_dataset, run_prediction_batched
from thermompnn.inference.siamese_inference import load_siamese_dataset, run_prediction_siamese

from thermompnn.trainer.v1_trainer import TransferModelPL
from thermompnn.trainer.v2_trainer import TransferModelPLv2
from thermompnn.trainer.siamese_trainer import TransferModelSiamesePL

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def get_metrics_full():
    return {
        "r2": R2Score().to('cuda'),
        "mse": MeanSquaredError(squared=True).to('cuda'),
        "rmse": MeanSquaredError(squared=False).to('cuda'),
        "spearman": SpearmanCorrCoef().to('cuda'),
        "pearson":  PearsonCorrCoef().to('cuda'),
    }


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20, surface_threshold: int = 15, backup_atom: str = "C", chain: str = 'A') -> torch.Tensor:
    """Expects ProteinMPNN feature dict and returns torch tensor of num_neighbors integers."""
    coords = xyz[basis_atom + f'_chain_{chain}']
    coords = torch.tensor(coords)
    # Compute distances and number of neighbors.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1

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

            out.append(-torch.unsqueeze(pred, 0))
        return out, log_probs


def inference(cfg, args):
    """Catch-all inference function for all ThermoMPNN versions"""

    # pre-initialization params
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ds_name = cfg.dataset
    model_name = cfg.name 
    
    if cfg.version == 'v1':
        # load specified dataset
        ds = load_v1_dataset(cfg)
        # load model weights
        model = TransferModelPL.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        # run inference function 
        if args.keep_preds:
            print('Keeping preds!')
            results = run_prediction_keep_preds(model_name, model, ds_name, ds, None, centrality=args.centrality)
        else:
            results = run_prediction_default(model_name, model, ds_name, ds, None)


    elif cfg.version == 'v2':
        ds = load_v2_dataset(cfg)
        model = TransferModelPLv2.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        results = run_prediction_batched(model_name, model, ds_name, ds, None, False)

    elif cfg.version == 'siamese':
        ds = load_siamese_dataset(cfg)
        model = TransferModelSiamesePL.load_from_checkpoint(args.model, cfg=cfg, map_location=device).model
        results = run_prediction_siamese(model_name, model, ds_name, ds, results, keep=False, use_both=True)

    else:
        raise ValueError("Invalid ThermoMPNN version specified. Options are v1, v2, siamese.")

    model = model.eval()
    model = model.cuda()

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"ThermoMPNN_{cfg.version}_{ds_name}_metrics.csv")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file for model and dataset params', type=str, default='../config.yaml')
    parser.add_argument('--model', help='path to model weights file (pt file)', type=str, default='checkpoint.ckpt')
    
    parser.add_argument('--keep_preds', action='store_true', default=False, 
                        help='Save raw model predictions as csv (currently supported for v1 only)')
    parser.add_argument('--centrality', action='store_true', default=False,
                        help='Calculate centrality value for each residue (# neighbors). Only used if --keep_preds is enabled.')

    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.load("../local.yaml"))
    with torch.no_grad():
        inference(cfg, args)