import argparse
import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import time
from torch.utils.data import DataLoader

from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.train_thermompnn import parse_cfg

from protein_mpnn_utils import alt_parse_PDB
from thermompnn.datasets.v2_datasets import tied_featurize_mut
from thermompnn.datasets.dataset_utils import Mutation
from thermompnn.model.v2_model import batched_index_select, _dist


def get_config(mode):
    if mode == 'single' or mode == 'additive':
        config = {
            'model':
            {
                'hidden_dims': [64, 32], 
                'subtract_mut': True, 
                'num_final_layers': 2,
                'freeze_weights': True, 
                'load_pretrained': True, 
                'lightattn': True
            }, 
            'platform': 
            {
                # 'thermompnn_dir': '/home/hdieckhaus/scripts/ThermoMPNN/'
                'thermompnn_dir': os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            }
        }
    elif mode == 'epistatic':
        config = {
            'model':
            {
                'hidden_dims': [128, 128], 
                'subtract_mut': False,
                'single_target': True, 
                'mutant_embedding': True,
                'num_final_layers': 2,
                'freeze_weights': True, 
                'load_pretrained': True, 
                'lightattn': True, 
                'aggregation': 'siamese', 
                'dropout': 0.1, 
                'edges': True
            }, 
            'platform': 
            {
                'thermompnn_dir': os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            }
        }

    config = OmegaConf.create(config)
    return parse_cfg(config)


def get_ssm_mutations_double(pdb):
    # make mutation list for SSM run
    mutation_list = []
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    MUT_POS, MUT_WT = [], []
    for seq_pos in range(len(pdb['seq'])):
        wtAA = pdb['seq'][seq_pos]
        # check for missing residues
        if wtAA != '-':
            MUT_POS.append(seq_pos)
            MUT_WT.append(wtAA)
        else:
            MUT_WT.append('-')

    # get all positional combos except self-combos
    from itertools import product
    pos_combos = [p for p in product(MUT_POS, MUT_POS) if p[0] != p[1]]
    pos_combos = np.array(pos_combos) # [combos, 2]
    wtAA = np.zeros_like(pos_combos)
    # fill in wtAA for each pos combo
    for p_idx in range(pos_combos.shape[0]):
        wtAA[p_idx, 0] = ALPHABET.index(MUT_WT[pos_combos[p_idx, 0]])
        wtAA[p_idx, 1] = ALPHABET.index(MUT_WT[pos_combos[p_idx, 1]])

    # make default mutAA bundle for broadcasting
    one = np.arange(20).repeat(20)
    two = np.tile(np.arange(20), 20)
    mutAA = np.stack([one, two]).T # [400, 2]

    n_comb = pos_combos.shape[0]
    # tile each to match [combos * 400, 2] shape
    mutAA = np.tile(mutAA, (n_comb, 1))
    wtAA = np.tile(wtAA, (400, 1))
    pos_combos = np.tile(pos_combos, (400, 1))

    # filter out self-mutations and single-mutations
    mask = np.sum(mutAA == wtAA, -1).astype(bool)
    pos_combos = pos_combos[~mask, :]
    mutAA = mutAA[~mask, :]
    wtAA = wtAA[~mask, :]
    return torch.tensor(pos_combos), torch.tensor(wtAA), torch.tensor(mutAA)


def run_double(all_mpnn_hid, mpnn_embed, cfg, loader, args, model, X, mask, mpnn_edges):
    """Batched mutation processing using shared protein embeddings and only stability prediction module head"""
    device = 'cuda'
    all_mpnn_hid = torch.cat(all_mpnn_hid[:cfg.model.num_final_layers], -1)
    embeds_all = [all_mpnn_hid, mpnn_embed]

    mpnn_embed = torch.cat(embeds_all, -1)
    mpnn_embed = mpnn_embed.repeat(args.batch_size, 1, 1)
    mpnn_edges = mpnn_edges.repeat(args.batch_size, 1, 1, 1)
    preds = []
    for b in tqdm(loader):
        pos, wtAA, mutAA = b
        pos = pos.to(device)
        wtAA = wtAA.to(device)
        mutAA = mutAA.to(device)
        mut_mutant_AAs = mutAA
        mut_wildtype_AAs = wtAA
        mut_positions = pos
                        
        # # preds += list(torch.squeeze(ddg, dim=-1).detach().cpu())

        # if enabled, get sequence embedding for mutant aa
        mut_embed_list = []
        for m in range(mut_mutant_AAs.shape[-1]):
            mut_embed = model.prot_mpnn.W_s(mut_mutant_AAs[:, m])
            mut_embed_list.append(mut_embed)
        mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)

        # get edges between the two mutated residues
        # E_idx is [B, K, L] and is a tensor of indices in X that should match neighbors of each residue
        D_n, E_idx = _dist(X[:, :, 1, :], mask)

        all_mpnn_edges = []
        n_mutations = [a for a in range(mut_positions.shape[-1])]
        for n_current in n_mutations:  # iterate over N-order mutations
            print(mpnn_edges.shape, mut_positions.shape)
            # select the edges at the current mutated positions
            mpnn_edges_tmp = torch.squeeze(batched_index_select(mpnn_edges, 1, mut_positions[:, n_current:n_current+1]), 1)
            E_idx_tmp = torch.squeeze(batched_index_select(E_idx, 1, mut_positions[:, n_current:n_current+1]), 1)

        #         # find matches for each position in the array of neighbors, grab edges, and add to list
        #         edges = []
        #         for b in range(E_idx_tmp.shape[0]):
        #             # iterate over all neighbors for each sample
        #             n_other = [a for a in n_mutations if a != n_current]
        #             tmp_edges = []
        #             for n_o in n_other:
        #                 idx = torch.where(E_idx_tmp[b, :] == mut_positions[b, n_o:n_o+1].expand(1, E_idx_tmp.shape[-1]))
        #                 if len(idx[0]) == 0: # if no edge exists, fill with empty edge for now
        #                     edge = torch.full([mpnn_edges_tmp.shape[-1]], torch.nan, device=E_idx.device)
        #                 else:
        #                     edge = mpnn_edges_tmp[b, idx[1][0], :]
        #                 tmp_edges.append(edge)

        #             # impute an empty edge if no neighbors exist
        #             try:
        #                 tmp_edges = torch.stack(tmp_edges, dim=-1)
        #             except RuntimeError: # if no neighbors exist, impute an empty edge
        #                 edge_shape = [self.HIDDEN_DIM, 1]
        #                 tmp_edges = torch.zeros(edge_shape, dtype=torch.float32, device=E_idx.device)

        #             # aggregate when multiple edges are returned (take mean of valid edges)
        #             edge = torch.nanmean(tmp_edges, dim=-1)
        #             edge = torch.nan_to_num(edge, nan=0)
        #             edges.append(edge)

        #         edges_compiled = torch.stack(edges, dim=0)
        #         all_mpnn_edges.append(edges_compiled)

        #     mpnn_edges = torch.stack(all_mpnn_edges, dim=-1) # shape: (Batch, Embed, N_muts)

        # # gather final representation from seq and structure embeddings
        # final_embed = [] 
        # for i in range(mut_mutant_AAs.shape[-1]):
        #     # gather embedding for a specific position
        #     current_positions = mut_positions[:, i:i+1] # [B, 1]
        #     g_struct_embed = torch.gather(all_mpnn_hid, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), all_mpnn_hid.size(2)))
        #     g_struct_embed = torch.squeeze(g_struct_embed, 1) # [B, E * nfl]
        #     # add specific mutant embedding to gathered embed based on which mutation is being gathered
        #     g_seq_embed = torch.gather(wt_embed, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), wt_embed.size(2)))
        #     g_seq_embed = torch.squeeze(g_seq_embed, 1) # [B, E]
        #     # if mut embed enabled, subtract it from the wt embed directly to keep dims low
        #     if self.cfg.model.mutant_embedding:
        #         g_seq_embed = g_seq_embed - mut_embed[:, :, i] # [B, E]
        #     g_embed = torch.cat([g_struct_embed, g_seq_embed], -1) # [B, E * (nfl + 1)]

        #     # if edges enabled, concatenate them onto the end of the embedding
        #     if self.cfg.model.edges:
        #         g_edge_embed = mpnn_edges[:, :, i]
        #         g_embed = torch.cat([g_embed, g_edge_embed], -1) # [B, E * (nfl + 2)]

        #     final_embed.append(g_embed)  # list with length N_mutations - used to make permutations
        # final_embed = torch.stack(final_embed, dim=0) # [2, B, E x (nfl + 1)]

        # # do initial dim reduction
        # final_embed = self.light_attention(final_embed) # [2, B, E]

        # # if batch is only single mutations, pad it out with a "zero" mutation
        # if final_embed.shape[0] == 1:
        #     zero_embed = torch.zeros(final_embed.shape, dtype=torch.float32, device=E_idx.device)
        #     final_embed = torch.cat([final_embed, zero_embed], dim=0)

        # # if batch is double-padded, check and zero out any singles in second half of embedding
        # else:
        #     single_mask = ~torch.logical_and(mut_wildtype_AAs[..., -1] == 0, mut_mutant_AAs[..., -1] == 0) # [B, ] 0 if single, 1 if double
        #     # print(single_mask.sum() / single_mask.numel()) # should be ~2:1 single:double (0.33 or so on average)
        #     single_mask = single_mask[..., None].expand(-1, final_embed.shape[-1])
        #     final_embed[-1, ...] = final_embed[-1, ...] * single_mask # zero out singles padded in Emb2
        
        # # make two copies, one with AB order and other with BA order of mutation
        # embedAB = torch.cat((final_embed[0, :, :], final_embed[1, :, :]), dim=-1)
        # embedBA = torch.cat((final_embed[1, :, :], final_embed[0, :, :]), dim=-1)

        # ddG_A = self.ddg_out(embedAB) # [B, 1]
        # ddG_B = self.ddg_out(embedBA) # [B, 1]

        # # multi-target mode has 21x21 = 441 output nodes, each corresponding to a combination of mutAA1 and mutAA2
        # if not self.cfg.model.single_target:
        #     # retrieve multi-target outputs using combined mutant positioning
        #     mutant_AA_idx = mut_mutant_AAs[:, 0] * 21 + mut_mutant_AAs[:, 1]
        #     ddG_mutA = torch.gather(ddG_A, 1, mutant_AA_idx[:, None])

        #     mutant_AA_idx = mut_mutant_AAs[:, 1] * 21 + mut_mutant_AAs[:, 0]
        #     ddG_mutB = torch.gather(ddG_B, 1, mutant_AA_idx[:, None])
        
        #     # if enabled, subtract predicted WT stability from each
        #     if self.cfg.model.subtract_mut:
        #         wt_AA_idx = mut_wildtype_AAs[:, 0] * 21 + mut_wildtype_AAs[:, 1]
        #         ddG_A = ddG_mutA - torch.gather(ddG_A, 1, wt_AA_idx[:, None])
        #         wt_AA_idx = mut_wildtype_AAs[:, 1] * 21 + mut_wildtype_AAs[:, 0]
        #         ddG_B = ddG_mutB - torch.gather(ddG_B, 1, wt_AA_idx[:, None])
        #     else:
        #         ddG_A = ddG_mutA
        #         ddG_B = ddG_mutB

        # return ddG_A, ddG_B


    return preds


class SSMDataset(torch.utils.data.Dataset):
    def __init__(self, POS, WTAA, MUTAA):
        self.POS = POS
        self.WTAA = WTAA
        self.MUTAA = MUTAA

    def __len__(self):
        return self.POS.shape[0]
    
    def __getitem__(self, index):

        return self.POS[index, :], self.WTAA[index, :], self.MUTAA[index, :]


def run_single_ssm(args, cfg, model):
    """Runs single-mutant SSM sweep with ThermoMPNN v2"""

    model.eval()
    model.cuda()
    stime = time.time()

    # parse PDB
    if args.chain is not None:
        chains = [c for c in args.chain]
    else:
        chains = None

    pdb = alt_parse_PDB(args.pdb, input_chain_list=chains)
    pdb[0]['mutation'] = Mutation([0], ['A'], ['A'], [0.], '') # placeholder mutation to keep featurization from throwing error

    # featurize input
    device = 'cuda'
    batch = tied_featurize_mut(pdb)
    X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch

    X = X.to(device)
    S = S.to(device)
    mask = mask.to(device)
    lengths = torch.Tensor(lengths).to(device)
    chain_M = chain_M.to(device)
    chain_encoding_all = chain_encoding_all.to(device)
    residue_idx = residue_idx.to(device)
    mut_ddGs = mut_ddGs.to(device)

    # do single pass through thermompnn
    X = torch.nan_to_num(X, nan=0.0)
    all_mpnn_hid, mpnn_embed, _, mpnn_edges = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)

    all_mpnn_hid = torch.cat(all_mpnn_hid[:cfg.model.num_final_layers], -1)
    all_mpnn_hid = torch.squeeze(torch.cat([all_mpnn_hid, mpnn_embed], -1), 0) # [L, E]

    all_mpnn_hid = model.light_attention(torch.unsqueeze(all_mpnn_hid, -1))

    ddg = model.ddg_out(all_mpnn_hid) # [L, 21]

    # subtract wildtype ddgs to normalize
    S = torch.squeeze(S) # [L, ]

    wt_ddg = batched_index_select(ddg, dim=-1, index=S) # [L, 1]
    ddg = ddg - wt_ddg.expand(-1, 21) # [L, 21]
    etime = time.time()
    elapsed = etime - stime
    pdb_name = os.path.basename(args.pdb)
    length = ddg.shape[0]
    print(f'ThermoMPNN single mutant predictions generated for protein {pdb_name} of length {length} in {round(elapsed, 2)} seconds.')
    return ddg, S


def expand_additive(ddg):
    """Uses torch broadcasting to add all possible single mutants to each other in a vectorized operation."""

    stime = time.time()
    # ddg [L, 21]
    dims = ddg.shape
    ddgA = ddg.reshape(dims[0], dims[1], 1, 1) # [L, 21, 1, 1]
    ddgB = ddg.reshape(1, 1, dims[0], dims[1]) # [1, 1, L, 21]
    ddg = ddgA + ddgB # L, 21, L, 21

    # mask out diagonal representing two mutations at the same position - this is invalid
    for i in range(dims[0]):
        ddg[i, :, i, :] = torch.nan

    etime = time.time()
    elapsed = etime - stime
    print(f'ThermoMPNN double mutant additive model predictions generated in {round(elapsed, 2)} seconds.')

    return ddg


def format_output_single(ddg, S, outname):
    """Converts raw SSM predictions into nice format for analysis"""
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    ddg = ddg.cpu().detach().numpy()
    L, AA = ddg.shape

    # make csv
    mutlist, ddglist = [], []
    for pos in tqdm(range(L)):
        wtAA = ALPHABET[S[pos]]
        for aa in range(AA):
            aa_name = ALPHABET[aa]
            ddg_single = ddg[pos, aa]
            mutlist.append(wtAA + str(pos + 1) + aa_name)
            ddglist.append(ddg_single)

    df = pd.DataFrame({
        'ddG (kcal/mol)': ddglist, 
        'Mutation': mutlist
    })
    df.to_csv(outname + '.csv')
    return 


def format_output_double(ddg, S, outname):
    """Converts raw SSM predictions into nice format for analysis"""
    stime = time.time()
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    ddg = ddg.cpu().detach().numpy() # [L, 21, L, 21]
    # L, AA = ddg.shape[:2]    
    L, AA = ddg.shape
    
    # this takes ~1min to reformat for csv saving (58M points) - how to speed this up?
    aa1, aa2 = np.meshgrid(np.arange(AA) , np.arange(AA)) # [441, ]
    aa1, aa2 = aa1.repeat(L).repeat(L), aa2.repeat(L).repeat(L) # [441 * L, ]

    pos1, pos2 = np.meshgrid(np.arange(L), np.arange(L))
    pos1, pos2 = pos1.repeat(AA).repeat(AA), pos2.repeat(AA).repeat(AA)
    ddglist, mutlist = [], []
    wt_seq = [ALPHABET[S[ppp]] for ppp in np.arange(L)]

    for a1, a2, p1, p2 in tqdm(zip(aa1, aa2, pos1, pos2)):
        pred = ddg[p1, a1] + ddg[p2, a2]
        if pred is torch.nan:
            continue
        ddglist.append(pred)

        wtAA1 = wt_seq[p1]
        wtAA2 = wt_seq[p2]
        mutAA1 = ALPHABET[a1]
        mutAA2 = ALPHABET[a2]
        mutname = wtAA1 + str(p1 + 1) + mutAA1 + ':' + wtAA2 + str(p2 + 1) + mutAA2
        mutlist.append(mutname)

    df = pd.DataFrame({
        'ddG (kcal/mol)': ddglist, 
        'Mutation': mutlist
    })
    df = df.dropna(subset=['ddG (kcal/mol)'])
    df.to_csv(outname + '.csv')
    etime = time.time()
    elapsed = etime - stime
    print(f'ThermoMPNN double mutant additive model predictions generated in {round(elapsed, 2)} seconds.')
    return


def run_epistatic_ssm(args, cfg, model):
    """Run epistatic model on double mutations """

    model.eval()
    model.cuda()
    stime = time.time()

    # parse PDB
    if args.chain is not None:
        chains = [c for c in args.chain]
    else:
        chains = None

    pdb = alt_parse_PDB(args.pdb, input_chain_list=chains)
    pdb[0]['mutation'] = Mutation([0], ['A'], ['A'], [0.], '') # placeholder mutation to keep featurization from throwing error

    # featurize input
    device = 'cuda'
    batch = tied_featurize_mut(pdb)
    X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch

    X = X.to(device)
    S = S.to(device)
    mask = mask.to(device)
    lengths = torch.Tensor(lengths).to(device)
    chain_M = chain_M.to(device)
    chain_encoding_all = chain_encoding_all.to(device)
    residue_idx = residue_idx.to(device)
    mut_ddGs = mut_ddGs.to(device)

    # do single pass through thermompnn
    X = torch.nan_to_num(X, nan=0.0)
    all_mpnn_hid, mpnn_embed, _, mpnn_edges = model.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)

    # grab double mutation inputs
    MUT_POS, MUT_WT_AA, MUT_MUT_AA = get_ssm_mutations_double(pdb[0])
    
    dataset = SSMDataset(MUT_POS, MUT_WT_AA, MUT_MUT_AA)
    loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=8)

    preds = run_double(all_mpnn_hid, mpnn_embed, cfg, loader, args, model, X, mask, mpnn_edges)

    quit()
    return

def main(args):

    cfg = get_config(args.mode)

    if args.mode == 'single' or args.mode == 'additive':
        cwd = os.path.dirname(os.path.realpath(__file__))
        cwd = os.path.dirname(os.path.dirname(cwd))
        model_path = os.path.join(cwd, 'model_weights/ThermoMPNN-ens1.ckpt')
        model = TransferModelPLv2.load_from_checkpoint(model_path, cfg=cfg).model
        
        # output: [L, 21]
        ddg, S = run_single_ssm(args, cfg, model)

        if args.mode == 'additive':
            format_output_double(ddg, S, args.out)
        else:
            format_output_single(ddg, S, args.out)
    elif args.mode == 'epistatic':
        cwd = os.path.dirname(os.path.realpath(__file__))
        cwd = os.path.dirname(os.path.dirname(cwd))
        model_path = os.path.join(cwd, 'model_weights/ThermoMPNN-D-ens1.ckpt')
        model = TransferModelPLv2Siamese.load_from_checkpoint(model_path, cfg=cfg).model

        ddg, S = run_epistatic_ssm(args, cfg, model)

    else:
        raise ValueError

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='SSM mode to use (single | additive | epistatic)', default='single')
    parser.add_argument('--pdb', type=str, help='PDB file to run', default='./2OCJ.pdb')
    parser.add_argument('--batch_size', type=int, help='batch size for stability prediction module', default=256)
    parser.add_argument('--out', type=str, help='output mutation prefix to save csv', default='tmp.csv')
    parser.add_argument('--chain', type=str, help='chain(s) to use (default is None, will use all chains)', default=None)
    main(parser.parse_args())
