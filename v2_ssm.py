import argparse
import os
import torch
import numpy as np
import re
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist


from thermompnn.datasets.v2_datasets import tied_featurize_mut
from thermompnn.datasets.dataset_utils import Mutation
from thermompnn.model.v2_model import batched_index_select, _dist
from thermompnn.ssm_utils import get_chains, get_config, custom_parse_PDB, idx_to_pdb_num, get_model, load_pdb


def get_ssm_mutations_double(pdb, dthresh):
    # make mutation list for SSM run
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

    # Use distance filter BEFORE data setup / inference for speedup
    from thermompnn.ssm_utils import get_dmat
    dmat = np.triu(get_dmat(pdb)) # [L, L]
    mask = (dmat < dthresh) & (dmat > 0.0)
    pos1, pos2 = np.where(mask)
    pos_combos = [(p1, p2) for p1, p2 in zip(pos1, pos2)]
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
    mutAA = np.tile(mutAA, (n_comb, 1))

    # the problem is 2nd wtAA/pos_combos and 2nd mutAA are correlated so they always show up together
    # repeat these 20x20 times
    wtAA = np.repeat(wtAA, 400, axis=0)
    pos_combos = np.repeat(pos_combos, 400, axis=0)

    # filter out self-mutations and single-mutations
    mask = np.sum(mutAA == wtAA, -1).astype(bool)
    pos_combos = pos_combos[~mask, :]
    mutAA = mutAA[~mask, :]
    wtAA = wtAA[~mask, :]

    # filter out upper-triangle portions - if mutAA or pos is larger, it's already been checked
    mask = pos_combos[:, 0] > pos_combos[:, 1]
    pos_combos = pos_combos[~mask, :]
    mutAA = mutAA[~mask, :]
    wtAA = wtAA[~mask, :]

    return torch.tensor(pos_combos), torch.tensor(wtAA), torch.tensor(mutAA)


def run_double(all_mpnn_hid, mpnn_embed, cfg, loader, args, model, X, mask, mpnn_edges_raw):
    """Batched mutation processing using shared protein embeddings and only stability prediction module head"""
    device = 'cuda'
    all_mpnn_hid = torch.cat(all_mpnn_hid[:cfg.model.num_final_layers], -1)
    all_mpnn_hid = all_mpnn_hid.repeat(args.batch_size, 1, 1)
    mpnn_embed = mpnn_embed.repeat(args.batch_size, 1, 1)
    mpnn_edges_raw = mpnn_edges_raw.repeat(args.batch_size, 1, 1, 1)
    # get edges between the two mutated residues
    D_n, E_idx = _dist(X[:, :, 1, :], mask)
    E_idx = E_idx.repeat(args.batch_size, 1, 1)

    preds = []
    for b in tqdm(loader):
        pos, wtAA, mutAA = b
        pos = pos.to(device)
        wtAA = wtAA.to(device)
        mutAA = mutAA.to(device)
        mut_mutant_AAs = mutAA
        mut_positions = pos
        REAL_batch_size = mutAA.shape[0]
                        
        # get sequence embedding for mutant aa
        mut_embed_list = []
        for m in range(mut_mutant_AAs.shape[-1]):
            mut_embed_list.append(model.prot_mpnn.W_s(mut_mutant_AAs[:, m]))
        mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)

        n_mutations = [0, 1]
        edges = []
        for n_current in n_mutations:  # iterate over N-order mutations
            # select the edges at the current mutated positions
            if REAL_batch_size != mpnn_edges_raw.shape[0]: # last batch will throw error if not corrected
                mpnn_edges_raw = mpnn_edges_raw[:REAL_batch_size, ...]
                E_idx = E_idx[:REAL_batch_size, ...]
                all_mpnn_hid = all_mpnn_hid[:REAL_batch_size, ...]
                mpnn_embed = mpnn_embed[:REAL_batch_size, ...]

            mpnn_edges_tmp = torch.squeeze(batched_index_select(mpnn_edges_raw, 1, mut_positions[:, n_current:n_current+1]), 1)
            E_idx_tmp = torch.squeeze(batched_index_select(E_idx, 1, mut_positions[:, n_current:n_current+1]), 1)

            n_other = [a for a in n_mutations if a != n_current]
            mp_other = mut_positions[:, n_other] # [B, 1]
            # E_idx_tmp [B, K]
            mp_other = mp_other[..., None].repeat(1, 1, E_idx_tmp.shape[-1]) # [B, 1, 48]
            idx = torch.where(E_idx_tmp[:, None, :] == mp_other) # get indices where the neighbor list matches the mutations we want
            a, b, c = idx
            # start w/empty edges and fill in as you go, then set remaining edges to 0
            edge = torch.full([REAL_batch_size, mpnn_edges_tmp.shape[-1]], torch.nan, device=E_idx.device) # [B, 128]
            # idx is (a, b, c) tuple of tensors
            # a has indices of batch members; b is all 0s; c has indices of actual neighbors for edge grabbing
            edge[a, :] = mpnn_edges_tmp[a, c, :]
            edge = torch.nan_to_num(edge, nan=0)
            edges.append(edge)

        mpnn_edges = torch.stack(edges, dim=-1) # this should get two edges per set of doubles (one for each)

        # gather final representation from seq and structure embeddings
        final_embed = [] 
        for i in range(mut_mutant_AAs.shape[-1]):
            # gather embedding for a specific position
            current_positions = mut_positions[:, i:i+1] # [B, 1]
            g_struct_embed = torch.gather(all_mpnn_hid, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), all_mpnn_hid.size(2)))
            g_struct_embed = torch.squeeze(g_struct_embed, 1) # [B, E * nfl]
            # add specific mutant embedding to gathered embed based on which mutation is being gathered
            g_seq_embed = torch.gather(mpnn_embed, 1, current_positions.unsqueeze(-1).expand(current_positions.size(0), current_positions.size(1), mpnn_embed.size(2)))
            g_seq_embed = torch.squeeze(g_seq_embed, 1) # [B, E]
            # if mut embed enabled, subtract it from the wt embed directly to keep dims low
            if cfg.model.mutant_embedding:
                if REAL_batch_size != mut_embed.shape[0]:
                    mut_embed = mut_embed[:REAL_batch_size, ...]
                g_seq_embed = g_seq_embed - mut_embed[:, :, i] # [B, E]
            g_embed = torch.cat([g_struct_embed, g_seq_embed], -1) # [B, E * (nfl + 1)]

            # if edges enabled, concatenate them onto the end of the embedding
            if cfg.model.edges:
                g_edge_embed = mpnn_edges[:, :, i]
                g_embed = torch.cat([g_embed, g_edge_embed], -1) # [B, E * (nfl + 2)]
            final_embed.append(g_embed)  # list with length N_mutations - used to make permutations
        final_embed = torch.stack(final_embed, dim=0) # [2, B, E x (nfl + 1)]

        # do initial dim reduction
        final_embed = model.light_attention(final_embed) # [2, B, E]

        # if batch is only single mutations, pad it out with a "zero" mutation
        if final_embed.shape[0] == 1:
            zero_embed = torch.zeros(final_embed.shape, dtype=torch.float32, device=E_idx.device)
            final_embed = torch.cat([final_embed, zero_embed], dim=0)

        # make two copies, one with AB order and other with BA order of mutation
        embedAB = torch.cat((final_embed[0, :, :], final_embed[1, :, :]), dim=-1)
        embedBA = torch.cat((final_embed[1, :, :], final_embed[0, :, :]), dim=-1)

        ddG_A = model.ddg_out(embedAB) # [B, 1]
        ddG_B = model.ddg_out(embedBA) # [B, 1]

        ddg = (ddG_A + ddG_B) / 2.
        preds += list(torch.squeeze(ddg, dim=-1).detach().cpu().numpy())
    return np.squeeze(preds)


class SSMDataset(torch.utils.data.Dataset):
    def __init__(self, POS, WTAA, MUTAA):
        self.POS = POS
        self.WTAA = WTAA
        self.MUTAA = MUTAA

    def __len__(self):
        return self.POS.shape[0]
    
    def __getitem__(self, index):

        return self.POS[index, :], self.WTAA[index, :], self.MUTAA[index, :]


def run_single_ssm(pdb, cfg, model):
    """Runs single-mutant SSM sweep with ThermoMPNN v2"""

    model.eval()
    model.cuda()
    stime = time.time()
    
    # placeholder mutation to keep featurization from throwing error
    pdb['mutation'] = Mutation([0], ['A'], ['A'], [0.], '')

    # featurize input
    device = 'cuda'
    batch = tied_featurize_mut([pdb])
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
    length = ddg.shape[0]
    print(f'ThermoMPNN single mutant predictions generated for protein of length {length} in {round(elapsed, 2)} seconds.')
    return ddg, S


def expand_additive(ddg):
    """Uses torch broadcasting to add all possible single mutants to each other in a vectorized operation."""
    # ddg [L, 21]
    dims = ddg.shape
    ddgA = ddg.reshape(dims[0], dims[1], 1, 1) # [L, 21, 1, 1]
    ddgB = ddg.reshape(1, 1, dims[0], dims[1]) # [1, 1, L, 21]
    ddg = ddgA + ddgB # L, 21, L, 21

    # mask out diagonal representing two mutations at the same position - this is invalid
    for i in range(dims[0]):
        ddg[i, :, i, :] = torch.nan

    return ddg


def format_output_single(ddg, S, threshold=-0.5):
    """Converts raw SSM predictions into nice format for analysis"""
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    ddg = ddg.cpu().detach().numpy()
    ddg = ddg[:, :20]

    keep_L, keep_AA = np.where(ddg <= threshold)
    ddg = ddg[ddg <= threshold] # [N, ]

    mutlist = []
    for L_idx, AA_idx in tqdm(zip(keep_L, keep_AA)):
        wtAA = ALPHABET[S[L_idx]]
        mutAA = ALPHABET[AA_idx]
        mutlist.append(wtAA + str(L_idx + 1) + mutAA)

    return ddg, mutlist


def format_output_double(ddg, S, threshold=-0.5):
    """Converts raw SSM predictions into nice format for analysis"""
    stime = time.time()
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    ddg = ddg.cpu().detach().numpy() # [L, 21]
    L, AA = ddg.shape

    ddg = expand_additive(ddg) # [L, 21, L, 21]
    ddg = ddg[:, :20, :, :20] # drop X predictions

    p1s, a1s, p2s, a2s = np.where(ddg <= threshold) # filter by threshold
    cond = (p1s < p2s) # & (a1s < a2s) # filter to keep only upper triangle
    p1s, a1s, p2s, a2s = p1s[cond], a1s[cond], p2s[cond], a2s[cond]
    wt_seq = [ALPHABET[S[ppp]] for ppp in np.arange(L)]

    mutlist, ddglist = [], []
    for p1, a1, p2, a2 in tqdm(zip(p1s, a1s, p2s, a2s)):
        wt1, wt2 = wt_seq[p1], wt_seq[p2]
        mut1, mut2 = ALPHABET[a1], ALPHABET[a2]

        if (wt1 != mut1) and (wt2 != mut2): # drop self-mutations
            mutation = f'{wt1}{p1 + 1}{mut1}:{wt2}{p2 + 1}{mut2}'
            mutlist.append(mutation)
            ddglist.append(ddg[p1, a1, p2, a2])

    etime = time.time()
    elapsed = etime - stime
    print(f'ThermoMPNN double mutant additive model predictions calculated in {round(elapsed, 2)} seconds.')
    return ddglist, mutlist


def format_output_epistatic(ddg, S, pos, wtAA, mutAA, threshold=-0.5):
    "Converts raw SSM predictions into nice format for analysis."
    stime = time.time()
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
    S = torch.squeeze(S)

    # filter out ddgs that miss the threshold
    mask = ddg <= threshold
    ddg = ddg[mask]
    wtAA = wtAA[mask, :]
    mutAA = mutAA[mask, :]
    pos = pos[mask, :]
    mut_list = []
    # a bunch of repeats in here?!
    for b in tqdm(range(ddg.shape[0])):
        w1 = ALPHABET[wtAA[b, 0]]
        w2 = ALPHABET[wtAA[b, 1]]
        m1 = ALPHABET[mutAA[b, 0]]
        m2 = ALPHABET[mutAA[b, 1]]
        mut_name = f'{w1}{pos[b, 0] + 1}{m1}:{w2}{pos[b, 1] + 1}{m2}'
        mut_list.append(mut_name)
    etime = time.time()
    elapsed = etime - stime
    print(f'ThermoMPNN double mutant epistatic model predictions sorted and filtered in {round(elapsed, 2)} seconds.')
    return ddg, mut_list


def run_epistatic_ssm(pdb, cfg, model, 
                      distance, threshold, batch_size):
    """Run epistatic model on double mutations """

    model.eval()
    model.cuda()
    stime = time.time()

    # placeholder mutation to keep featurization from throwing error
    pdb['mutation'] = Mutation([0], ['A'], ['A'], [0.], '') 

    # featurize input
    device = 'cuda'
    batch = tied_featurize_mut([pdb])
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
    MUT_POS, MUT_WT_AA, MUT_MUT_AA = get_ssm_mutations_double(pdb, distance)
    # dataset = SSMDataset(MUT_POS, MUT_WT_AA, MUT_MUT_AA)
    # loader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=8)

    # preds = run_double(all_mpnn_hid, mpnn_embed, cfg, loader, args, model, X, mask, mpnn_edges)
    # ddg, mutations = format_output_epistatic(preds, S, MUT_POS, MUT_WT_AA, MUT_MUT_AA, args.threshold)
    
    # etime = time.time()
    # elapsed = etime - stime
    # print(f'ThermoMPNN double mutant epistatic model predictions generated in {round(elapsed, 2)} seconds.')
    # return ddg, mutations



def renumber_pdb(df, args):
    """Renumber output mutations to match PDB numbering for interpretation"""
    # parse PDB
    chains = get_chains(args.pdb, args.chains)
    pdb = custom_parse_PDB(args.pdb, input_chain_list=chains)[0]

    if (args.mode == 'additive') or (args.mode == 'epistatic'):
        # grab positions
        df[['mut1', 'mut2']] = df['Mutation'].str.split(':', n=2, expand=True)
        df['pos1'] = df['mut1'].str[1:-1].astype(int) - 1
        df['pos2'] = df['mut2'].str[1:-1].astype(int) - 1

        df['pos1'] = idx_to_pdb_num(pdb, df['pos1'].values)
        df['pos2'] = idx_to_pdb_num(pdb, df['pos2'].values)

        df['wt1'], df['wt2'] = df['mut1'].str[0], df['mut2'].str[0]
        df['mt1'], df['mt2'] = df['mut1'].str[-1], df['mut2'].str[-1]

        df['Mutation'] = df['wt1'] + df['pos1'] + df['mt1'] + ':' + df['wt2'] + df['pos2'] + df['mt2']
        df = df[['ddG (kcal/mol)', 'Mutation', 'CA-CA Distance']].reset_index(drop=True)

    else:
        # grab position
        df['pos'] = df['Mutation'].str[1:-1].astype(int) - 1

        df['pos'] = idx_to_pdb_num(pdb, df['pos'].values)
        df['wt'] = df['Mutation'].str[0]
        df['mt'] = df['Mutation'].str[-1]

        df['Mutation'] = df['wt'] + df['pos'] + df['mt']   
        df = df[['ddG (kcal/mol)', 'Mutation']].reset_index(drop=True)

    print(f'ThermoMPNN predictions renumbered.')
    return df 


def disulfide_penalty(df, pdb_file, chain_list, model):
  """Automatically detects disulfide breakage based on Cys-Cys distance."""

  chain_list = get_chains(pdb_file, chain_list)
  pdb_dict = custom_parse_PDB(pdb_file, input_chain_list=chain_list, side_chains=True)

  # collect all SG coordinates from all chains
  coords_all = [k for k in pdb_dict[0].keys() if k.startswith('coords')]
  chains = [c[-1] for c in coords_all]
  sg_coords = [pdb_dict[0][c][f'SG_chain_{chain}'] for c, chain in zip(coords_all, chains)]
  sg_coords = np.concatenate(sg_coords, axis=0)

  # calculate pairwise distance and threshold to find disulfides
  dist = cdist(sg_coords, sg_coords)
  dist = np.nan_to_num(dist, 10000)
  hits = np.where((dist < 3) & (dist > 0)) # tuple of two [N] arrays of indices

  if model == 'single':
    df['wtAA'] = df['Mutation'].str[0]
    df['mutAA'] = df['Mutation'].str[-1]
    df['pos'] = df['Mutation'].str[1:-1].astype(int) - 1

    # match hit indices to actual resns for penalty
    bad_resns = []
    for h in hits[0]:
      bad_resns.append(h)

    print('Identified the following disulfide engaged residues:', bad_resns)

    # apply penalty
    penalty = 2  # in kcal/mol - higher is less stable
    mask = df['pos'].isin(bad_resns) & (df['wtAA'] != df['mutAA'])

    df.loc[mask, 'ddG (kcal/mol)'] = df.loc[mask, 'ddG (kcal/mol)'] + penalty
    return df[['Mutation', 'ddG (kcal/mol)']].reset_index(drop=True)

  else:
    df[['mut1', 'mut2']] = df['Mutation'].str.split(':', n=2, expand=True)
    df['wtAA1'] = df['mut1'].str[0]
    df['mutAA1'] = df['mut1'].str[-1]
    df['pos1'] = df['mut1'].str[1:-1].astype(int) - 1

    df['wtAA2'] = df['mut2'].str[0]
    df['mutAA2'] = df['mut2'].str[-1]
    df['pos2'] = df['mut2'].str[1:-1].astype(int) - 1

    bad_resns = []
    for h in hits[0]:
      bad_resns.append(h)

    print('Identified the following disulfide engaged residues:', bad_resns)

    # apply penalty
    penalty = 2  # in kcal/mol - higher is less stable
    mask = df['pos1'].isin(bad_resns) & (df['wtAA1'] != df['mutAA1'])
    mask2 = df['pos2'].isin(bad_resns) & (df['wtAA2'] != df['mutAA2'])
    mask = mask | mask2

    df.loc[mask, 'ddG (kcal/mol)'] = df.loc[mask, 'ddG (kcal/mol)'] + penalty
    return df[['Mutation', 'ddG (kcal/mol)', 'CA-CA Distance']].reset_index(drop=True)


def main(args):

    cfg = get_config(args.mode)
    model = get_model(args.mode, cfg)
    pdb_data = load_pdb(args.pdb, args.chains)
    pdbname = os.path.basename(args.pdb)
    print(f'Loaded PDB {pdbname}')

    if (args.mode == 'single') or (args.mode == 'additive'):
        ddg, S = run_single_ssm(pdb_data, cfg, model)

        if args.mode == 'single':
            ddg, mutations = format_output_single(ddg, S, args.threshold)
        else:
            ddg, mutations = format_output_double(ddg, S, args.threshold)

    elif args.mode == 'epistatic':
        ddg, mutations = run_epistatic_ssm(pdb_data, cfg, model, 
                                           args.distance, args.threshold, args.batch_size)

    else:
        raise ValueError

    df = pd.DataFrame({
        'ddG (kcal/mol)': ddg, 
        'Mutation': mutations
    })

    # if args.mode != 'single':
        # df = distance_filter(df, args)
    
    if args.ss_penalty:
        df = disulfide_penalty(df, args.pdb, args.chains, model=args.mode)

    df = df.dropna(subset=['ddG (kcal/mol)'])
    if args.threshold <= -0.:
        df = df.sort_values(by=['ddG (kcal/mol)'])

    if args.mode != 'single': # sort to have same output order
        df[['mut1', 'mut2']] = df['Mutation'].str.split(':', n=2, expand=True)
        df['pos1'] = df['mut1'].str[1:-1].astype(int) + 1
        df['pos2'] = df['mut2'].str[1:-1].astype(int) + 1

        df = df.sort_values(by=['pos1', 'pos2'])
        df = df[['ddG (kcal/mol)', 'Mutation', 'CA-CA Distance']].reset_index(drop=True)

    try:
      df = renumber_pdb(df, args)
    except (KeyError, IndexError):
      print('PDB renumbering failed (sorry!) You can still use the raw position data. Or, you can renumber your PDB, fill any weird gaps, and try again.')
    
    df.to_csv(args.out + '.csv')

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='SSM mode to use (single | additive | epistatic)', default='single')
    parser.add_argument('--pdb', type=str, help='PDB file to run', default='./2OCJ.pdb')
    parser.add_argument('--batch_size', type=int, help='batch size for stability prediction module', default=256)
    parser.add_argument('--out', type=str, help='output mutation prefix to save csv', default='ssm')
    parser.add_argument('--chains', nargs='+', help='chain(s) to use. Default is None, which will use all chains. Example: A B C')
    parser.add_argument('--threshold', type=float, default=-0.5, help='Threshold for SSM sweep. By default, ThermoMPNN only saves mutations below this threshold (-0.5 kcal/mol). To save all mutations, set this really high (e.g., 100)')
    parser.add_argument('--distance', type=float, default=10.0, help='Filter for double mutant predictions using pairwise Ca distance cutoff (default is 5 A).')
    parser.add_argument('--ss_penalty', action='store_true', help='Add explicit disulfide breakage penalty. Default is False.')
    main(parser.parse_args())
