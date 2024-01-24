import torch
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from tqdm import tqdm
from copy import deepcopy
from itertools import permutations
import itertools

from thermompnn.protein_mpnn_utils import alt_parse_PDB, parse_PDB
from thermompnn.datasets.dataset_utils import Mutation, seq1_index_to_seq2_index, ALPHABET


def tied_featurize_mut(batch, device='cpu', chain_dict=None, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None,
                   pssm_dict=None, bias_by_res_dict=None, ca_only=False):
    """ Pack and pad batch into torch tensors - modified to also handle mutation data"""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    else:
        X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([B, L_max], dtype=np.float32)  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([B, L_max, 21], dtype=np.float32)  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones([B, L_max, 21],
                                          dtype=np.float32)  # 1.0 for the bits that need to be predicted
    chain_M_pos = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([B, L_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    S = np.zeros([B, L_max], dtype=np.int32)
    omit_AA_mask = np.zeros([B, L_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []

    for i, b in enumerate(batch):
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[
                b['name']]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
            visible_chains = []
        num_chains = b['num_of_chains']
        all_chains = masked_chains + visible_chains

    # mutation info packed into batched tensors
    N_MUT = max([len(b['mutation'].position) for b in batch])  # use max mut for padding
    # N_MUT = len(batch[0]['mutation'].position) # different size depending on mutation count (single, double, etc.)
    MUT_POS = np.zeros([B, N_MUT], dtype=np.int32)  # position of a given mutation
    MUT_WT_AA = np.zeros([B, N_MUT], dtype=np.int32)  # WT amino acid
    MUT_MUT_AA = np.zeros([B, N_MUT], dtype=np.int32)  # Mutant amino acid
    MUT_DDG = np.zeros([B, 1], dtype=np.float32)  # Mutation ddG (WT ddG - Mutant ddG)
    
    for i, b in enumerate(batch):
        mask_dict = {}
        a = 0
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_AA_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}'])  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([chain_coords[c] for c in
                                        [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                         f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                if ca_only:
                    x_chain = np.array(chain_coords[f'CA_chain_{letter}'])  # [chain_lenght,1,3] #CA_diff
                    if len(x_chain.shape) == 2:
                        x_chain = x_chain[:, None, :]
                else:
                    x_chain = np.stack([chain_coords[c] for c in
                                        [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                         f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict != None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_AA_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_AA_dict != None:
                    for item in omit_AA_dict[b['name']][letter]:
                        idx_AA = np.array(item[0]) - 1
                        AA_idx = np.array([np.argwhere(np.array(list(alphabet)) == AA)[0][0] for AA in item[1]]).repeat(
                            idx_AA.shape[0])
                        idx_ = np.array([[a, b] for a in idx_AA for b in AA_idx])
                        omit_AA_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_AA_mask_list.append(omit_AA_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(L_max)
        if tied_positions_dict != None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                set_chains_tied = set(list(itertools.chain(*[list(item) for item in tied_pos_list])))
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        if isinstance(v[0], list):
                            for v_count in range(len(v[0])):
                                one_list.append(start_idx + v[0][v_count] - 1)  # make 0 to be the first
                                tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
                        else:
                            for v_ in v:
                                one_list.append(start_idx + v_ - 1)  # make 0 to be the first
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(fixed_position_mask_list, 0)  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list, 0)  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list, 0)  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list, 0)  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list,
                                      0)  # [L,21], 0.0 for places where AA frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        
        # NOTE: this causes size mismatches b/c it pads the seq dim as well as the length dim
        # omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list, 0), [[0, L_max - l]], 'constant', constant_values=(0.0,))
        omit_AA_mask_pad = np.pad(np.concatenate(omit_AA_mask_list, 0), [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        
        chain_M[i, :] = m_pad
        chain_M_pos[i, :] = m_pos_pad
        omit_AA_mask[i,] = omit_AA_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, L_max - l]], 'constant', constant_values=(0.0,))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0, L_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)
        
        # retrieving mutant info as a vector
        mut = b['mutation']
        MUT_DDG[i, :] = mut.ddG
        # keep ALL mutation info (handles multiple mutations)
        positions = mut.position
        wtAAs = [ALPHABET.index(aa) for aa in mut.wildtype]
        mutAAs = [ALPHABET.index(aa) for aa in mut.mutation]
        for nm in range(len(positions)):  # handle variable number of mutations (zero padded)
            MUT_POS[i, nm] = positions[nm]
            MUT_WT_AA[i, nm] = wtAAs[nm]
            MUT_MUT_AA[i, nm] = mutAAs[nm]

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    pssm_coef_all = torch.from_numpy(pssm_coef_all).to(dtype=torch.float32, device=device)
    pssm_bias_all = torch.from_numpy(pssm_bias_all).to(dtype=torch.float32, device=device)
    pssm_log_odds_all = torch.from_numpy(pssm_log_odds_all).to(dtype=torch.float32, device=device)

    tied_beta = torch.from_numpy(tied_beta).to(dtype=torch.float32, device=device)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = torch.from_numpy(bias_by_res_all).to(dtype=torch.float32, device=device)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate([phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1)  # [B,L,3]
    dihedral_mask = torch.from_numpy(dihedral_mask).to(dtype=torch.float32, device=device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_M_pos = torch.from_numpy(chain_M_pos).to(dtype=torch.float32, device=device)
    omit_AA_mask = torch.from_numpy(omit_AA_mask).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    if ca_only:
        X_out = X[:, :, 0]
    else:
        X_out = X

    # convert to tensor here
    MUT_POS = torch.from_numpy(MUT_POS).to(dtype=torch.long, device=device)
    MUT_WT_AA = torch.from_numpy(MUT_WT_AA).to(dtype=torch.long, device=device)
    MUT_MUT_AA = torch.from_numpy(MUT_MUT_AA).to(dtype=torch.long, device=device)
    MUT_DDG = torch.from_numpy(MUT_DDG).to(dtype=torch.float32, device=device)

    return X_out, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, MUT_POS, MUT_WT_AA, MUT_MUT_AA, MUT_DDG


def batchify(lengths, batch_size=10000):
    """Batchify a set of sequence lengths into padded batches of max size batch_size"""
    # adapted from proteinmpnn.utils.StructureLoader

    # argsort returns indexes, not sorted list
    sorted_ix = np.argsort(lengths)
    # Cluster into batches of similar sizes
    clusters, batch = [], []
    batch_max = 0
    for ix in sorted_ix:
        size = lengths[ix]
        if size * (len(batch) + 1) <= batch_size:  # make sure new size (B x L_max) is under batch_size
            batch.append(ix)
            batch_max = size
        else:
            clusters.append(np.array(batch))
            batch, batch_max = [ix], size
        
    if len(batch) > 0:
        clusters.append(np.array(batch))
    # clusters is a list of (lists of indexes which each make up a batch with length <= self.batch_size)
    return clusters


class MegaScaleDatasetv2(torch.utils.data.Dataset):
    """Rewritten Megascale dataset doing batched mutation generation"""

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split  # which split to retrieve
        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)

        # new type-specific data loading - add option for multi-mutations
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)
        mut_list = []

        if 'single' in self.cfg.data.mut_types:
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))
        if 'double' in self.cfg.data.mut_types:
            mut_list.append(df.loc[(df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :].reset_index(drop=True))

        if len(mut_list) == 0:  # special case of loading rev muts w/no fwd muts
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))

        self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)  # this includes points missing structure data

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "all": [], 
        }
        
        if self.split == 'all':
            all_names = splits['train'] + splits['val'] + splits['test']
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]
        # filter df for only data with structural data
        self.df = self.df.loc[self.df.WT_name.isin(self.wt_names)].reset_index(drop=True)
        
        df_list = []
        # pick which mutations to use (data augmentation)
        if 'single' in self.cfg.data.mut_types:
            print('Including %s direct single mutations' % str(self.df.shape[0]))
            df_list.append(self.df)
        elif 'double' in self.cfg.data.mut_types:
            print('Including %s direct double mutations' % str(self.df.shape[0]))
            df_list.append(self.df)

        if 'reverse' in cfg.data.mut_types:
            print('Adding reverse mutations!')
            # add reverse mutations to CSV dataset
            flipped_df = self._add_reverse_mutations()
            print('Generated %s reverse mutations' % str(flipped_df.shape[0]))
            df_list.append(flipped_df)
        
        if 'permutations' in cfg.data.mut_types:
            print('Adding permuted mutations!')
            permuted_df = self._add_permuted_mutations()
            print('Generated %s permuted mutations' % str(permuted_df.shape[0]))
            df_list.append(permuted_df)

        self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)
        print('Final Dataset Size: %s ' % str(self.df.shape[0]))


        # generate batches (lists of df idx for pulling data and matching with PDBs)
        print('Batch size:', self.cfg.training.batch_size)
        self.clusters = batchify(self.df.aa_seq.str.len().values, self.cfg.training.batch_size)
        print('Generated %s batches of size %s for %s split' % (str(len(self.clusters)), str(self.cfg.training.batch_size), self.split))

        # load ALL pdb data only once
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.split(".pdb")[0].replace("|",":")
            pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file)
            self.pdb_data[wt_name] = pdb[0]

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is list of protein-mutation pairs (can be different proteins)."""
        batch_idx = self.clusters[index]
        pdb_list = []
        for i, row in self.df.iloc[batch_idx].iterrows():
            # load PDB and correct seq as needed
            pdb_CANONICAL = self.pdb_data[row.WT_name.strip('.pdb')]
            pdb = deepcopy(pdb_CANONICAL)  # avoid modifying the original PDB sequence, just in case
            mut_types = row.mut_type.split(':')
            wt_list, mut_list, idx_list = [], [], []
            
            # if len(mut_types) > 1:  # for additive model testing
                # mut_types = [mut_types[0]]

            for mt in mut_types:
                wt = mt[0]
                mut = mt[-1]
                pos = int(mt[1:-1]) - 1
                # seq needs to reflect mutation, even if flipped/modified from WT structure
                if pdb['seq'][pos] != wt:
                    # NEED to modify the seq_chain_A field in addition to the full seq field
                    seq_keys = [k for k in pdb.keys() if k.startswith('seq')]
                    if len(seq_keys) > 2:
                        raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
                    for sk in seq_keys:
                        tmp = [p for p in pdb[sk]]
                        tmp[pos] = wt
                        pdb[sk] = ''.join(tmp)
                    # check that both seqs got changed
                    assert pdb[seq_keys[0]] == pdb[seq_keys[1]]
                
                wt_list.append(wt)
                mut_list.append(mut)
                idx_list.append(pos)
            ddG = -1 * float(row.ddG_ML)
                        
            pdb['mutation'] = Mutation(idx_list, wt_list, mut_list, ddG, row.WT_name)
            tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point
            pdb_list.append(tmp)

        # putting tied_featurize here means the CPU, not the GPU, handles it, and it is parallelized to each DataLoader
        features = tied_featurize_mut(pdb_list, 'cpu')
        return features
    
    def _add_reverse_mutations(self):
        # for each mutation, add row w/opposite ddG sign + flipped seq, wt, mut, etc.
        flipped_df = self.df.copy(deep=True)
        # grab flipped mut/wt - DO NOT just reverse the string, this will break the numbering
        flipped_df['mut_type'] = flipped_df['mut_type'].str[-1] + flipped_df['mut_type'].str[1:-1] + flipped_df['mut_type'].str[0]
        flipped_df['ddG_ML'] = flipped_df['ddG_ML'].astype(float) * -1
        # flip WT sequence value as well
        flipped_df['pos'] = flipped_df['mut_type'].str[1:-1].astype(int) - 1
        flipped_df['mut'] = flipped_df['mut_type'].str[-1]
        
        def flip_string(x):
            tmp = [s for s in x['aa_seq']]
            tmp[x['pos']] = x['mut']
            return ''.join(tmp)
        
        flipped_df['aa_seq'] = flipped_df.apply(lambda x: flip_string(x), axis=1)
        flipped_df = flipped_df.drop(labels=['pos', 'mut'], axis='columns')
        return flipped_df

    def _add_permuted_mutations(self):
        
        def permute_subset(chunk):
            """generate permutations of a given df chunk"""
            # check only one wt AA is present
            chunk['wt'] = chunk['mut_type'].str[0]
            pos = int(chunk['pos'].unique()[0])
            wt_name = chunk['WT_name'].unique()[0]
            assert len(chunk['wt'].unique()) == 1
            # retrieve all mutant AAs
            chunk['mut'] = chunk['mut_type'].str[-1]
            
            # generate every possible permutation of N mutations (N * (N-1) combinations)
            all_combos = [p for p in permutations(iter(chunk['mut']), r=2)]
            # sometimes there is no combo
            if len(all_combos) < 5:
                return pd.DataFrame()
            wt, mut = zip(*all_combos)

            # unpack and calulate new ddGs
            aa_seqs = []
            
            new_df = pd.DataFrame({'wt': wt, 'mut': mut})
            new_df['mut_type'] = new_df.wt + str(pos) + new_df.mut

            # correct aa seq to reflect new values
            aa_seq = chunk.aa_seq.values[0]
            
            aa_seqs = []
            for t in new_df['wt']:
                tmp = [p for p in aa_seq]
                tmp[pos - 1] = t[0]
                aa_seq = ''.join(tmp)
                aa_seqs.append(aa_seq)
                assert aa_seq[pos - 1] == t[0]

            new_df['aa_seq'] = aa_seqs
            
            # get aa-to-index values from chunk df
            wt = chunk['mut']
            ix = chunk.index
            decode = {}
            for w, i in zip(wt, ix):
                decode[w] = i
            
            wt_idx_list = [decode[aa] for aa in new_df['wt'].values]
            mut_idx_list = [decode[aa] for aa in new_df['mut'].values]

            # do vectorize ddG selection
            ddg1 = chunk.loc[wt_idx_list]['ddG_ML'].astype(float).values
            ddg2 = chunk.loc[mut_idx_list]['ddG_ML'].astype(float).values
            # new "mutant" sign stays the same, but new "wt" needs to be flipped since it was mutant before
            ddgs = ddg2 - ddg1
            new_df['ddG_ML'] = ddgs
            new_df['pos'] = pos
            new_df['WT_name'] = wt_name
            return new_df
        
        permuted_df = self.df.copy(deep=True)
        
        permuted_df['pos'] = permuted_df['mut_type'].str[1:-1].astype(int)
        permuted_df = permuted_df.groupby(["WT_name", "pos"], group_keys=False).apply(lambda x: permute_subset(x))
        
        return permuted_df


class ddgBenchDatasetv2(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname, flip=False):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        self.rev = flip  # "reverse" mutation testing
        print('Reverse mutations: %s' % str(self.rev))
        df = pd.read_csv(csv_fname)
        self.df = df

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()
                 
        self.pdb_data = {}
        
        # parse all PDBs first - treat each row as its own PDB
        for i, row in self.df.iterrows():
            fname = row.PDB[:-1]
            pdb_file = os.path.join(self.pdb_dir, f"{fname}.pdb")
            chain = [row.PDB[-1]]
            pdb = alt_parse_PDB(pdb_file, chain)
            self.pdb_data[i] = pdb[0]

    def __len__(self):
        return len(self.pdb_data)

    def __getitem__(self, index):
        """Batch retrieval fxn - do each row as its own item, for simplicity"""

        pdb_list = []
        row = self.df.iloc[index]
        pdb_CANONICAL = self.pdb_data[index]
        
        if 'MUTS' in self.df.columns:
            mut_info = row.MUTS
            # mut_info = row.MUT1
            # mut_info = row.MUT10
        else:
            mut_info = row.MUT

        wt_list, mut_list, idx_list = [], [], []
        if mut_info is np.nan:  # to skip missing mutations for additive model
            return

        if (self.cfg.dataset == 'ptmul') and ('double' not in self.cfg.data.mut_types):
            if len(mut_info.split(';')) < 3: # skip double mutants if missing
                return
        if (self.cfg.dataset == 'ptmul') and ('higher' not in self.cfg.data.mut_types):
            if len(mut_info.split(';')) > 2: # skip higher order mutants if missing
                return

        
        if not self.rev:
            for mt in mut_info.split(';'):  # handle multiple mutations like for megascale
                
                wtAA, mutAA = mt[0], mt[-1]
                ddG = float(row.DDG) * -1
                
                pdb = deepcopy(pdb_CANONICAL)      
                pdb_idx = self._get_pdb_idx(mt, pdb_CANONICAL)
                assert pdb['seq'][pdb_idx] == wtAA
                
                wt_list.append(wtAA)
                mut_list.append(mutAA)
                idx_list.append(pdb_idx)
                pdb['mutation'] = Mutation(idx_list, wt_list, mut_list, ddG, row.PDB[:-1])
                
        else:  # reverse mutations - retrieve Rosetta/modeled structures
            fname = row.PDB[:-1]
            chain = row.PDB[-1]
            old_idx = int(mut_info[1:-1])
            # print(f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")
            pdb_file = os.path.join(self.pdb_dir, f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")
            pdb = alt_parse_PDB(pdb_file, chain)[0]
            pdb['mutation'] = Mutation([pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])
        
        # if needed, update wt seq to match passed mutations
        seq_keys = [k for k in pdb.keys() if k.startswith('seq')]
        if len(seq_keys) > 2:
            raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
        for sk in seq_keys:
            tmp = [p for p in pdb[sk]]
            tmp[pdb_idx] = wtAA
            pdb[sk] = ''.join(tmp)   

        tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point
        pdb_list.append(tmp)
        features = tied_featurize_mut(pdb_list, 'cpu')
        return features

        # ------ Old version set up for only single mutants ----- #
        # wtAA, mutAA = mut_info[0], mut_info[-1]
        # pdb_idx = self._get_pdb_idx(mut_info, pdb_CANONICAL)
        # ddG = float(row.DDG) * -1

        # if not self.rev:
        #     pdb = deepcopy(pdb_CANONICAL)  # if you don't do this, it will overwrite each PDB entry if modified
        #     assert pdb['seq'][pdb_idx] == wtAA
        #     pdb['mutation'] = Mutation([pdb_idx], [wtAA], [mutAA], ddG, row.PDB[:-1])
            
        # else:  # retrieve Rosetta structure and save with reverse mutation
        #     fname = row.PDB[:-1]
        #     chain = row.PDB[-1]
        #     old_idx = int(mut_info[1:-1])
        #     # print(f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")
        #     pdb_file = os.path.join(self.pdb_dir, f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")
        #     pdb = alt_parse_PDB(pdb_file, chain)[0]
        #     pdb['mutation'] = Mutation([pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])
        
        # seq_keys = [k for k in pdb.keys() if k.startswith('seq')]
        # if len(seq_keys) > 2:
        #     raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
        # for sk in seq_keys:
        #     tmp = [p for p in pdb[sk]]
        #     tmp[pdb_idx] = wtAA
        #     pdb[sk] = ''.join(tmp)   

        # tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point
        # pdb_list.append(tmp)
        # features = tied_featurize_mut(pdb_list, 'cpu')
        # return features
    
    def _get_pdb_idx(self, mut_info, pdb):
        """Helper function to fix any alignment issues with messy experimental data"""
        wtAA = mut_info[0]
        try:
            pos = mut_info[1:-1]
            pdb_idx = pdb['resn_list'].index(pos)
            
        except ValueError:  # skip positions with insertion codes for now - hard to parse
            raise ValueError('NO PDB IDX FOUND - insertion code')
        try:
            assert pdb['seq'][pdb_idx] == wtAA
        except AssertionError:  # contingency for mis-alignments
            # if gaps are present, add these to idx (+10 to get any around the mutation site, very much an ugly hack)
            if 'S669' in self.pdb_dir:
                gaps = [g for g in pdb['seq'] if g == '-']
            elif ('PTMUL' in self.pdb_dir) and ('1YCC' in pdb['name']) and ('T69E' == mut_info):  # bad case w/negative bits
                gaps = ['-']
            elif ('PTMUL' in self.pdb_dir) and ('1QJP' in pdb['name'] and int(mut_info[1:-1]) > 150):
                gaps = ['-'] * 34

            elif ('PTMUL' in self.pdb_dir) and ('2WSY' in pdb['name']):  # this protein does not behave well
                gaps = ['-'] * 15

                if int(mut_info[1:-1]) == 232:
                    gaps = ['-'] * 29
                if int(mut_info[1:-1]) == 175:
                    gaps = ['-'] * 9
                if int(mut_info[1:-1]) == 209:
                    gaps = ['-'] * 28
                    
            else:
                gaps = [g for g in pdb['seq'][:pdb_idx + 10] if g == '-']                

            if len(gaps) > 0:
                pdb_idx += len(gaps)
            else:
                pdb_idx += 1

            if pdb_idx is None:
                raise ValueError('NO PDB IDX FOUND - bad alignment')
            
            assert pdb['seq'][pdb_idx] == wtAA
        return pdb_idx


class FireProtDatasetv2(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        filename = self.cfg.data_loc.fireprot_csv

        df = pd.read_csv(filename).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.df = df

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.fireprot_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "homologue-free": [],
            "all": []
        }

        if self.split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        self.df = self.df.loc[self.df.pdb_id_corrected.isin(self.wt_names)]
        print('Total dataset size:', self.df.shape)
        # this is a hack to get around different chain names in Fireprot dataset
        self.cfg.training.batch_size = 100
        self.clusters = batchify(self.df.pdb_sequence.str.len().values, self.cfg.training.batch_size)
        print('Generated %s batches of size %s for %s split' % (str(len(self.clusters)), str(self.cfg.training.batch_size), self.split))

        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.rstrip('.pdb')
            pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file)
            self.pdb_data[wt_name] = pdb[0]
        
    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, index):

        batch_idx = self.clusters[index]
        pdb_list = []
        for i, row in self.df.iloc[batch_idx].iterrows():
            # load PDB and correct seq as needed
            wt_name = row.pdb_id_corrected.rstrip('.pdb')
            pdb = self.pdb_data[wt_name]
    
            try:
                pdb_idx = row.pdb_position
                assert pdb['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
                
            except AssertionError:  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(row.pdb_sequence, pdb['seq'].replace("-", "X"))
                pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if pdb_idx is None:
                    continue
                assert pdb['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]

            ddG = float(row.ddG)
            mut = Mutation([pdb_idx], [pdb['seq'][pdb_idx]], [row.mutation], ddG, wt_name)
            
            pdb['mutation'] = mut
            tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point
            pdb_list.append(tmp)

        features = tied_featurize_mut(pdb_list, 'cpu')
        return features


class MegaScaleDatasetv2Aug(MegaScaleDatasetv2):
    """Rewritten Megascale dataset doing batched mutation generation"""

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split  # which split to retrieve
        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)

        # new type-specific data loading - add option for multi-mutations
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)
        mut_list = []
        if 'single' in self.cfg.data.mut_types:
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))
        
        if 'double' in self.cfg.data.mut_types:
            mut_list.append(df.loc[(df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :].reset_index(drop=True))

        if len(mut_list) == 0:  # special case of loading rev muts w/no fwd muts
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))

        self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)  # this includes points missing structure data

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "all": [], 
        }
        
        if self.split == 'all':
            all_names = splits['train'] + splits['val'] + splits['test']
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]
        # filter df for only data with structural data
        self.df = self.df.loc[self.df.WT_name.isin(self.wt_names)].reset_index(drop=True)
        
                
        df_list = []
        # pick which mutations to use (data augmentation)
        if 'single' in self.cfg.data.mut_types:
            print('Including %s direct single mutations' % str(self.df.shape[0]))
            self.df['DIRECT'] = True
            self.df['wt_orig'] = self.df['mut_type'].str[0]  # mark original WT for file loading use
            df_list.append(self.df)
            
        if 'double' in self.cfg.data.mut_types:
            print('Including %s direct double mutations' % str(self.df.shape[0]))
            self.df['DIRECT'] = True
            df_list.append(self.df)
        
        if 'reverse' in cfg.data.mut_types:
            print('Adding reverse mutations!')
            # add reverse mutations to CSV dataset
            flipped_df = self._add_reverse_mutations()
            print('Generated %s reverse mutations' % str(flipped_df.shape[0]))
            flipped_df['DIRECT'] = False
            df_list.append(flipped_df)
        
        if 'permutations' in cfg.data.mut_types:
            print('Adding permuted mutations!')
            permuted_df = self._add_permuted_mutations()
            print('Generated %s permuted mutations' % str(permuted_df.shape[0]))
            permuted_df['DIRECT'] = False
            df_list.append(permuted_df)

        if 'double-aug' in cfg.data.mut_types:
            # grab single mutants even if not included in mutation type list
            tmp = df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True)
            tmp = tmp.loc[tmp.WT_name.isin(self.wt_names)].reset_index(drop=True)
            
            aug_mutant_multiple = 1  # for each single mutant, how many double mutant augmented points to make
            double_aug = self._augment_double_mutants(tmp, n=aug_mutant_multiple)
            print('Generated %s augmented double mutations' % str(double_aug.shape[0]))
            double_aug['DIRECT'] = False
            df_list.append(double_aug)
        
        self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)
        print('Final Dataset Size: %s ' % str(self.df.shape[0]))

        # generate batches (lists of df idx for pulling data and matching with PDBs)
        self.clusters = batchify(self.df.aa_seq.str.len().values, self.cfg.training.batch_size)
        print('Generated %s batches of size %s for %s split' % (str(len(self.clusters)), str(self.cfg.training.batch_size), self.split))

        # pre-loading wildtype structures - can avoid file I/O for 50% of data points
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.split(".pdb")[0].replace("|",":")
            pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file)
            self.pdb_data[wt_name] = pdb[0]
    
    def _augment_double_mutants(self, df, n=1):
        """Use pairs of single mutants and modeled structures to make new double mutant data points
        df is the full dataframe (can be just single mutants or single + double)
        n is the number of augmented data points to make for every single mutant example"""
        
        print('Augment Mutant Coefficient:', n)
        new_df = df.copy(deep=True)
        # trim to only single mutants
        new_df = new_df.loc[~new_df.mut_type.str.contains(':')]
        new_df['position'] = new_df['mut_type'].str[1:-1]
        mutation_list, ddg_list = [], []
        
        # run stochastic augmentation on every single mutant
        for i, row in tqdm(new_df.iterrows()):
            # select a random row from the same protein with a different position
            aug_options = new_df.loc[(new_df.WT_name == row.WT_name) & (new_df.position != row.position)]
            # for each chosen sample, make a new data point
            chosen = aug_options.sample(n=n)
            for c, crow in chosen.iterrows():
                # combine mut type and ddG columns, all else can remain
                new_mut = row.mut_type[-1] + row.mut_type[1:-1] + row.mut_type[0]  # can't reverse string - breaks numbers
                new_mut = new_mut + ':' + crow.mut_type
                new_ddg = float(crow.ddG_ML) - float(row.ddG_ML)

                mutation_list.append(new_mut)
                ddg_list.append(new_ddg)
        
        new_df = new_df.drop('position', axis=1)
        df_list = []
        # for each mutation chunk, 
        for c in range(n):
            print(c, n)
            partial_df = new_df.copy(deep=True)
            # select every nth mutation with an offset of c
            partial_df['mut_type'] = mutation_list[c : : n]
            partial_df['ddG_ML'] = ddg_list[c : : n]
            df_list.append(partial_df)
            print(partial_df.head)
            print('-' * 50)
        print('=' * 50)
        new_df = pd.concat(df_list, axis=0)
        print(new_df.head)
        return new_df
    
    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is list of protein-mutation pairs (can be different proteins)."""
        batch_idx = self.clusters[index]
        pdb_list = []
        for i, row in self.df.iloc[batch_idx].iterrows():

            pdb_loc = '/work/users/d/i/dieckhau/rocklin_data/FINAL_results/'
            wt_name = row.WT_name.rstrip(".pdb").replace("|",":")
            chain = 'A'  # all Rocklin proteins have chain A, since they're simulated

            mt = row.mut_type  # only single mutations for now
            direct = row.DIRECT
            # need to retrieve file and compile mutation object
            
            # four options: single, rev, double, double-rev
            if len(mt.split(':')) > 1: # double
                wt_list, pos_list, mut_list = [], [], []
                for m in mt.split(':'):
                    wt_list.append(m[0])
                    pos_list.append(int(m[1:-1]) - 1)
                    mut_list.append(m[-1])
                if direct: # double
                    pdb = self.pdb_data[row.WT_name.removesuffix('.pdb')]
                    
                else: # double-rev
                    # need to flip the wt and mutant AAs for file retrieval
                    wt_ros = mut_list[0]
                    pos_ros = pos_list[0] + 1
                    mut_ros = wt_list[0]
                    pdb_file = os.path.join(pdb_loc, 
                                            wt_name, 
                                            'pdb_models', 
                                            f'{chain}[{wt_ros}{pos_ros}{mut_ros}].pdb')
                    assert os.path.isfile(pdb_file)  # check that file exists
                    pdb = parse_PDB(pdb_file)[0]
                    
            else:
                wt_list = [mt[0]]
                mut_list = [mt[-1]]
                pos_list = [int(mt[1:-1]) - 1]
                if direct: # single
                    pdb = self.pdb_data[row.WT_name.removesuffix('.pdb')]
                    
                else: # single-rev
                    pdb_file = os.path.join(pdb_loc, 
                                            wt_name, 
                                            'pdb_models', 
                                            f'{chain}[{mut_list[0]}{pos_list[0] + 1}{wt_list[0]}].pdb')
                    assert os.path.isfile(pdb_file)  # check that file exists
                    pdb = parse_PDB(pdb_file)[0]
                                    
            tmp_pdb = deepcopy(pdb) # this is hacky but it is needed or else it overwrites all PDBs with the last data point

            ddG = -1 * float(row.ddG_ML)
            tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, ddG, row.WT_name)
            pdb_list.append(tmp_pdb)

        # putting tied_featurize here means the CPU, not the GPU, handles it, and it is parallelized to each DataLoader
        features = tied_featurize_mut(pdb_list, 'cpu')
        # save as .pt file for later loading
        # loc = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/double_aug'
        # loc = '/proj/kuhl_lab/users/dieckhau/data/mega_scale/batched_TR_TP'
        # fpath = os.path.join('%s/%s' % (loc, self.split), f'batch_{index}.pt')
        # torch.save(features, fpath)
        # print('Saved batch %s' % str(index))
        return features


class MegaScaleDatasetv2Pt(torch.utils.data.Dataset):
    """Rewritten Megascale dataset loading individual .pt files as batches"""

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split  # which split to retrieve
        
        if 'reverse' in cfg.data.mut_types:
            if 'permutations' in cfg.data.mut_types:
                self.pt_loc = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/batched_TR_TP', self.split)
                
            else:
                self.pt_loc = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/batched_TR', self.split)
                
        elif 'confidence' in cfg.data.mut_types:
            print('Loading confidence batched dataset')
            self.pt_loc = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/conf_batches', self.split)
            
        elif 'double-aug' in cfg.data.mut_types:
            print('Loading augmented double mutant dataset')
            self.pt_loc = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/double_aug', self.split)
        else:
            raise ValueError("Invalid PT-based dataset selected")

        self.batch_files = sorted(os.listdir(self.pt_loc))
        self.batch_files = [sb for sb in self.batch_files if sb.endswith('.pt')]
            
    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is pre-packed into a .pt file"""
        current_batch = f'batch_{index}.pt'
        current_batch = os.path.join(self.pt_loc, current_batch)
        features = torch.load(current_batch)
        return features


class FireProtDatasetv2Confidence(torch.utils.data.Dataset):
    """For confidence estimation"""
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split  # which split to retrieve
        
        if 'confidence' in cfg.data.mut_types:
            print('Loading confidence batched dataset')
            self.pt_loc = os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/batched_mega_scale/conf_batches/fireprot', self.split)
        else:
            raise ValueError("Invalid PT-based dataset selected")

        self.batch_files = sorted(os.listdir(self.pt_loc))
        self.batch_files = [sb for sb in self.batch_files if sb.endswith('.pt')]
            
    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is pre-packed into a .pt file"""
        current_batch = f'batch_{index}.pt'
        current_batch = os.path.join(self.pt_loc, current_batch)
        features = torch.load(current_batch)
        return features


class MegaScaleDatasetv2Rebatched(MegaScaleDatasetv2Aug):
    """Rewritten Megascale dataset doing truly batched mutation generation (getitem returns 1 sample)"""

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split  # which split to retrieve
        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)

        # new type-specific data loading - add option for multi-mutations
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)
        mut_list = []
        if 'single' in self.cfg.data.mut_types:
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))
        
        if 'double' in self.cfg.data.mut_types:
            mut_list.append(df.loc[(df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :].reset_index(drop=True))

        if len(mut_list) == 0:  # special case of loading rev muts w/no fwd muts
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))

        self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)  # this includes points missing structure data

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        self.wt_names = splits[self.split]
        # filter df for only data with structural data
        self.df = self.df.loc[self.df.WT_name.isin(self.wt_names)].reset_index(drop=True)
         
        df_list = []
        # pick which mutations to use (data augmentation)
        if ('single' in self.cfg.data.mut_types) or ('double' in self.cfg.data.mut_types):
            print('Including %s direct single/double mutations' % str(self.df.shape[0]))
            self.df['DIRECT'] = True
            self.df['wt_orig'] = self.df['mut_type'].str[0]  # mark original WT for file loading use
            df_list.append(self.df)
            
        if 'double-aug' in cfg.data.mut_types:
            # grab single mutants even if not included in mutation type list
            tmp = df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True)
            tmp = tmp.loc[tmp.WT_name.isin(self.wt_names)].reset_index(drop=True)
            
            # derive c value as inverse of batch_fraction
            batch_c = 1 if 'batch_fraction' not in cfg.training else int(1/cfg.training.batch_fraction)
            
            double_aug = self._augment_double_mutants(tmp, c=batch_c)
            print('Generated %s augmented double mutations' % str(double_aug.shape[0]))
            double_aug['DIRECT'] = False
            df_list.append(double_aug)
        
        self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)

        # TODO rewrite batchify to reorder an existing self.df according to length
        self._sort_dataset()
        print('Final Dataset Size: %s ' % str(self.df.shape[0]))

        # pre-loading wildtype structures - can avoid later file I/O for 50% of data points
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.split(".pdb")[0].replace("|",":")
            pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file)
            self.pdb_data[wt_name] = pdb[0]
    
    def _sort_dataset(self):
        """Sort the main df by sequence length"""
        self.df['length'] = self.df.aa_seq.str.len()
        self.df = self.df.sort_values(by='length')
        self.df.drop(columns='length')
        return
    
    def __len__(self):
        """Total sample count instead of batch count"""
        return self.df.shape[0]
    
    def _augment_double_mutants(self, df, c=1):
        """Use pairs of single mutants and modeled structures to make new double mutant data points
        Rewritten to be vectorized  - can handle arbitrary multiplication ratios"""
        
        new_df = df.copy(deep=True)
        # trim to only single mutants
        new_df = new_df.loc[~new_df.mut_type.str.contains(':')]
        new_df['position'] = new_df['mut_type'].str[1:-1]
        mutation_list, ddg_list = [], []

        positions = new_df.position.values
        wtns = new_df.WT_name.values

        chunks, pchunks = {}, {}
        # do sweep for masks only ONCE
        for un in np.unique(wtns):
            chunk = wtns == un
            chunks[un] = chunk
        
        for pos in np.unique(positions):
            chunk = positions == pos
            pchunks[pos] = chunk

        mutations = new_df.mut_type.values
        ddgs = new_df.ddG_ML.values
        for p, w, m, d in tqdm(zip(positions, wtns, mutations, ddgs)):
            # pair must be in the same protein but different position, random selection
            mask = chunks[w] * ~pchunks[p]
            options = mutations[mask]
            ddgs_paired = ddgs[mask]

            chosen = np.random.choice(np.arange(options.size), size=c)
            
            for ch in chosen:
                new_ddg = ddgs_paired[ch]
                new_ddg = float(new_ddg) - float(d)

                new_mut = options[ch]
                new_mut = m[-1] + m[1:-1] + m[0] + ':' + new_mut

                mutation_list.append(new_mut)
                ddg_list.append(new_ddg)
        
        # parse out offset values
        tmp_df = new_df.copy(deep=True)
        df_list = []
        for c_i in range(c):
            mut_chunk = mutation_list[c_i::c]
            ddg_chunk = ddg_list[c_i::c]

            tmp_df['mut_type'] = mut_chunk
            tmp_df['ddG_ML'] = ddg_chunk
            df_list.append(tmp_df)

        new_df = pd.concat(df_list, axis=0)
        return new_df

    
    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is list of protein-mutation pairs (can be different proteins)."""
        row = self.df.iloc[index]

        pdb_loc = self.cfg.data_loc.rosetta_data
        # pdb_loc = '/work/users/d/i/dieckhau/rocklin_data/FINAL_results/'
        wt_name = row.WT_name.rstrip(".pdb").replace("|",":")
        chain = 'A'  # all Rocklin proteins have chain A, since they're simulated

        mt = row.mut_type  # only single mutations for now
        direct = row.DIRECT
        # need to retrieve file and compile mutation object
        
        # four options: single, rev, double, double-rev
        if len(mt.split(':')) > 1: # double
            wt_list, pos_list, mut_list = [], [], []
            for m in mt.split(':'):
                wt_list.append(m[0])
                pos_list.append(int(m[1:-1]) - 1)
                mut_list.append(m[-1])
            if direct: # double
                pdb = self.pdb_data[row.WT_name.removesuffix('.pdb')]
                
            else: # double-rev
                # need to flip the wt and mutant AAs for file retrieval
                wt_ros = mut_list[0]
                pos_ros = pos_list[0] + 1
                mut_ros = wt_list[0]

                pt_file = os.path.join(pdb_loc, wt_name, 'pdb_models', f'{chain}[{wt_ros}{pos_ros}{mut_ros}].pt')
                # if pt exists, it's way faster to load than using the pdb parser
                if os.path.isfile(pt_file):
                    pdb = torch.load(pt_file)
                else:
                    pdb_file = os.path.join(pdb_loc, wt_name, 'pdb_models', f'{chain}[{wt_ros}{pos_ros}{mut_ros}].pdb')
                    assert os.path.isfile(pdb_file)  # check that file exists
                    pdb = parse_PDB(pdb_file)[0]
                    torch.save(pdb, pt_file)
                
        else:
            wt_list = [mt[0]]
            mut_list = [mt[-1]]
            pos_list = [int(mt[1:-1]) - 1]
            if direct: # single
                pdb = self.pdb_data[row.WT_name.removesuffix('.pdb')]
                
            else: # single-rev
                pdb_file = os.path.join(pdb_loc, 
                                        wt_name, 
                                        'pdb_models', 
                                        f'{chain}[{mut_list[0]}{pos_list[0] + 1}{wt_list[0]}].pdb')
                assert os.path.isfile(pdb_file)  # check that file exists
                pdb = parse_PDB(pdb_file)[0]
                                
        tmp_pdb = deepcopy(pdb) # this is hacky but it is needed or else it overwrites all PDBs with the last data point

        ddG = -1 * float(row.ddG_ML)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, ddG, row.WT_name)
        # return a SINGLE pdb object this way
        return tmp_pdb


def prebatch_dataset(dataset, workers=1):
    """Runs pre-batching for large (augmented) datasets"""
    from torch.utils.data import DataLoader
    print('Starting Prebatching for dataset...')
    print('Number of workers:', workers)

    loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=workers, batch_size=None)
    
    for batch in tqdm(loader):
        pass
    
    return

if __name__ == "__main__":
    # testing functionality
    from omegaconf import OmegaConf
    import sys
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))
    split = sys.argv[3]
    
    prebatch_dataset(dataset=MegaScaleDatasetv2Rebatched(cfg, split),
                     workers=cfg.training.num_workers)
    
    # ds = MegaScaleDatasetv2Pt(cfg, 'test')

    # ds = MegaScaleDatasetv2(cfg, 'test')
    # ds = FireProtDatasetv2(cfg, 'test')
    # ds = ddgBenchDatasetv2(cfg, '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/protddg-bench-master/P53/pdbs', '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/protddg-bench-master/P53/p53_clean.csv')

    # print('Starting dataset iteration')
    # for batch in tqdm(ds):
    #     pass