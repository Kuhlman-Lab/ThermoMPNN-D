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
from Bio.PDB import PDBParser

from thermompnn.trainer.v2_trainer import TransferModelPLv2, TransferModelPLv2Siamese
from thermompnn.train_thermompnn import parse_cfg

from thermompnn.datasets.v2_datasets import tied_featurize_mut
from thermompnn.datasets.dataset_utils import Mutation
from thermompnn.model.v2_model import batched_index_select, _dist


def idx_to_pdb_num(pdb, poslist):
  # set up PDB resns and boundaries
  chains = [key[-1] for key in pdb.keys() if key.startswith('resn_list_')]
  resn_lists = [pdb[key] for key in pdb.keys() if key.startswith('resn_list')]
  converter = {}
  offset = 0
  for n, rlist in enumerate(resn_lists):
      chain = chains[n]
      for idx, resid in enumerate(rlist):
          converter[idx + offset] = chain + resid
      offset += idx + 1

  return [converter[pos] for pos in poslist]


def custom_parse_PDB_biounits(x, atoms=['N', 'CA', 'C'], chain=None):
    '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}

    def AA_to_N(x):
        # ["ARND"] -> [[0,1,2,3]]
        x = np.array(x);
        if x.ndim == 0: x = x[None]
        return [[aa_1_N.get(a, states - 1) for a in y] for y in x]

    def N_to_AA(x):
        # [[0,1,2,3]] -> ["ARND"]
        x = np.array(x);
        if x.ndim == 1: x = x[None]
        return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    resn_list = []
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()

        # handling MSE and SEC residues
        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")
        elif line[17:17 + 3] == "MSE":
            line = line.replace("MSE", "MET")
        elif line[17:17 + 3] == "SEC":
            line = line.replace("SEC", "CYS")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()

                # TODO check for gaps and add them if needed
                if (resn not in resn_list) and len(resn_list) > 0:
                  _, num, ins_code = re.split(r'(\d+)', resn)
                  _, num_prior, ins_code_prior = re.split(r'(\d+)', resn_list[-1])
                  gap = int(num) - int(num_prior) - 1
                  for g in range(gap + 1):
                    resn_list.append(str(int(num_prior) + g))

                # RAW resn is defined HERE
                resn_list.append(resn) # NEED to keep ins code here

                x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]
                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1
                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz[resn]:
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq[resn]:
                    seq[resn][resa] = resi

                if atom not in xyz[resn][resa]:
                    xyz[resn][resa][atom] = np.array([x, y, z])

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k], 20))
            else:
                seq_.append(20)

            if resn in xyz:
                for k in sorted(xyz[resn]):
                    for atom in atoms:
                        if atom in xyz[resn][k]:
                            xyz_.append(xyz[resn][k][atom])
                        else:
                            xyz_.append(np.full(3, np.nan))
            else:
                for atom in atoms: xyz_.append(np.full(3, np.nan))
        return np.array(xyz_).reshape(-1, len(atoms), 3), N_to_AA(np.array(seq_)), list(dict.fromkeys(resn_list))
    except TypeError:
        return 'no_chain', 'no_chain', 'no_chain'


def custom_parse_PDB(path_to_pdb, input_chain_list=None, ca_only=False, side_chains=False, mut_chain=None):
    c = 0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        concat_N = []
        concat_CA = []
        concat_C = []
        concat_O = []
        concat_mask = []
        coords_dict = {}
        for letter in chain_alphabet:
            if ca_only:
                sidechain_atoms = ['CA']
            elif side_chains:
                sidechain_atoms = ["N", "CA", "C", "O", "CB",
                                   "CG", "CG1", "OG1", "OG2", "CG2", "OG", "SG",
                                   "CD", "SD", "CD1", "ND1", "CD2", "OD1", "OD2", "ND2",
                                   "CE", "CE1", "NE1", "OE1", "NE2", "OE2", "NE", "CE2", "CE3",
                                   "NZ", "CZ", "CZ2", "CZ3", "CH2", "OH", "NH1", "NH2"]
            else:
                sidechain_atoms = ['N', 'CA', 'C', 'O']
            xyz, seq, resn_list = custom_parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
            if resn_list != 'no_chain':
              my_dict['resn_list_' + letter] = resn_list
                  # my_dict['resn_list'] = list(resn_list)
            if type(xyz) != str:
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                if ca_only:
                    coords_dict_chain['CA_chain_' + letter] = xyz.tolist()
                elif side_chains:
                    coords_dict_chain['SG_chain_' + letter] = xyz[:, 11].tolist()
                else:
                    coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                    coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                    coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                    coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1

        fi = biounit.rfind("/")
        # if mut_chain is None:
          # my_dict['resn_list'] = list(resn_list)
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        # my_dict['resn_list'] = list(resn_list)
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


def get_config(mode):
    """Grabs relevant configs from disk."""

    current_location = os.path.dirname(os.path.realpath(__file__))
    local = os.path.join(os.path.join(current_location), 'examples/configs/local.yaml')

    if mode == 'single' or mode == 'additive':
        aux = os.path.join(os.path.join(current_location), 'examples/configs/single.yaml')

    elif mode == 'epistatic':
        aux = os.path.join(os.path.join(current_location), 'examples/configs/epistatic.yaml')
    else:
        raise ValueError("Invalid mode selected!")

    config = OmegaConf.merge(
        OmegaConf.load(local), 
        OmegaConf.load(aux)
    )

    return parse_cfg(config)


def get_chains(pdb_file, chain_list):
  # collect list of chains in PDB to match with input
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('', pdb_file)
  pdb_chains = [c.id for c in structure.get_chains()]

  if chain_list is None: # fill in all chains if left blank
    chain_list = pdb_chains
  elif len(chain_list) < 1:
    chain_list = pdb_chains

  for ch in chain_list:
    assert ch in pdb_chains, f"Chain {ch} not found in PDB file with chains {pdb_chains}"

  return chain_list


def get_ssm_mutations_double(pdb):
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


def run_single_ssm(args, cfg, model):
    """Runs single-mutant SSM sweep with ThermoMPNN v2"""

    model.eval()
    model.cuda()
    stime = time.time()

    # parse PDB
    chains = get_chains(args.pdb, args.chains)

    pdb = custom_parse_PDB(args.pdb, input_chain_list=chains)
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
    L, AA = ddg.shape
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


def run_epistatic_ssm(args, cfg, model):
    """Run epistatic model on double mutations """

    model.eval()
    model.cuda()
    stime = time.time()

    # parse PDB
    chains = get_chains(args.pdb, args.chains)

    pdb = custom_parse_PDB(args.pdb, input_chain_list=chains)
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
    ddg, mutations = format_output_epistatic(preds, S, MUT_POS, MUT_WT_AA, MUT_MUT_AA, args.threshold)
    
    etime = time.time()
    elapsed = etime - stime
    print(f'ThermoMPNN double mutant epistatic model predictions generated in {round(elapsed, 2)} seconds.')
    return ddg, mutations


def distance_filter(df, args):
    """filter df based on pdb distances"""
    # parse PDB
    chains = get_chains(args.pdb, args.chains)

    pdb = custom_parse_PDB(args.pdb, input_chain_list=chains)[0]

    # grab positions
    df[['mut1', 'mut2']] = df['Mutation'].str.split(':', n=2, expand=True)
    df['pos1'] = df['mut1'].str[1:-1].astype(int) - 1
    df['pos2'] = df['mut2'].str[1:-1].astype(int) - 1

    # get distance matrix
    coords = [k for k in pdb.keys() if k.startswith('coords_chain_')]
    # compile all-by-all coords into big matrix
    coo_all = []
    for coord in coords:
      ch = coord.split('_')[-1]
      coo = np.stack(pdb[coord][f'CA_chain_{ch}']) # [L, 3]
      coo_all.append(coo)
    coo_all = np.concatenate(coo_all) # [L_total, 3]
    dmat = cdist(coo_all, coo_all)

    # filter df based on positions
    pos1, pos2 = df['pos1'].values, df['pos2'].values
    dist_list = []
    for p1, p2 in tqdm(zip(pos1, pos2)):
        dist_list.append(dmat[p1, p2])

    df['CA-CA Distance'] = dist_list
    df = df.loc[df['CA-CA Distance'] <= args.distance]
    df['CA-CA Distance'] = df['CA-CA Distance'].round(2)

    df = df[['ddG (kcal/mol)', 'Mutation', 'CA-CA Distance']].reset_index(drop=True)
    print(f'Distance matrix generated.')
    return df


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
    if (args.mode == 'single') or (args.mode == 'additive'):
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, 'model_weights/ThermoMPNN-ens1.ckpt')
        model = TransferModelPLv2.load_from_checkpoint(model_path, cfg=cfg).model
        
        # output: [L, 21]
        ddg, S = run_single_ssm(args, cfg, model)

        if args.mode == 'additive':
            ddg, mutations = format_output_double(ddg, S, args.threshold)
        else:
            ddg, mutations = format_output_single(ddg, S, args.threshold)
    elif args.mode == 'epistatic':
        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, 'model_weights/ThermoMPNN-D-ens1.ckpt')
        model = TransferModelPLv2Siamese.load_from_checkpoint(model_path, cfg=cfg).model

        ddg, mutations = run_epistatic_ssm(args, cfg, model)

    else:
        raise ValueError

    df = pd.DataFrame({
        'ddG (kcal/mol)': ddg, 
        'Mutation': mutations
    })

    if args.mode != 'single':
        df = distance_filter(df, args)
    
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
