import torch
from torch import nn
from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef, PearsonCorrCoef

import argparse
import omegaconf as OmegaConf
import pandas as pd

from thermompnn.model.modules import get_protein_mpnn
from thermompnn.protein_mpnn_utils import tied_featurize


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
