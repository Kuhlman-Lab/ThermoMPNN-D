import torch
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from Bio.SeqUtils import seq1
from tqdm import tqdm
from copy import deepcopy
from itertools import permutations
import itertools
from sklearn.preprocessing import minmax_scale

from thermompnn.protein_mpnn_utils import alt_parse_PDB, parse_PDB
from thermompnn.datasets.dataset_utils import Mutation, seq1_index_to_seq2_index, ALPHABET


def tied_featurize_mut(batch, device='cpu', chain_dict=None, fixed_position_dict=None, omit_AA_dict=None, tied_positions_dict=None,
                   pssm_dict=None, bias_by_res_dict=None, ca_only=False, side_chains=False, esm=False):
    """ Pack and pad batch into torch tensors - modified to also handle mutation data"""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    try:
        lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
    except TypeError:
        return None
    L_max = max([len(b['seq']) for b in batch])
    if ca_only:
        X = np.zeros([B, L_max, 1, 3])
    elif side_chains:
        X = np.zeros([B, L_max, 14, 3])
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

    # mutation info packed into batched tensors
    N_MUT = max([len(b['mutation'].position) for b in batch])  # use max mut for padding
    # N_MUT = len(batch[0]['mutation'].position) # different size depending on mutation count (single, double, etc.)
    MUT_POS = np.zeros([B, N_MUT], dtype=np.int32)  # position of a given mutation
    MUT_WT_AA = np.zeros([B, N_MUT], dtype=np.int32)  # WT amino acid
    MUT_MUT_AA = np.zeros([B, N_MUT], dtype=np.int32)  # Mutant amino acid
    MUT_DDG = np.zeros([B, 1], dtype=np.float32)  # Mutation ddG (WT ddG - Mutant ddG)
    if esm:
        ESM_embeds = np.zeros([B, L_max, 320], dtype=np.float32) # zero if padded

    for i, b in enumerate(batch):
        
        # NOTE: new chunk below
        if chain_dict != None:
            masked_chains, visible_chains = chain_dict[
                b['name']]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
            visible_chains = []
        all_chains = masked_chains + visible_chains
        # NOTE: new chunk above
        
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
                elif side_chains:
                    x_chain = np.stack([chain_coords[c] for c in chain_coords.keys()], 1) # [chain_length, 14, 3]
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
        if esm:
            ESM_embeds[i, :lengths[i], :] = b['esm'][1:-1]

    isnan = np.isnan(X)

    # need to protect against "missing" side chains, else whole sequence will be flagged
    if not side_chains:
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    else:
        mask = np.isfinite(np.sum(X[..., :4, :], (2, 3))).astype(np.float32)

    if not side_chains:
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
    atom_mask = torch.from_numpy(np.prod(isnan, axis=-1)).to(dtype=torch.long, device=device)
    if esm:
        ESM_embeds = torch.from_numpy(ESM_embeds).to(dtype=torch.float32, device=device)
        return X_out, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, MUT_POS, MUT_WT_AA, MUT_MUT_AA, MUT_DDG, atom_mask, ESM_embeds
    else:
        return X_out, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, MUT_POS, MUT_WT_AA, MUT_MUT_AA, MUT_DDG, atom_mask


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
        self.side_chains = self.cfg.data.get('side_chains', False)
        # parse all PDBs first - treat each row as its own PDB
        for i, row in self.df.iterrows():
            fname = row.PDB[:-1]
            pdb_file = os.path.join(self.pdb_dir, f"{fname}.pdb")
            chain = [row.PDB[-1]]
            pdb = alt_parse_PDB(pdb_file, input_chain_list=chain, side_chains=self.side_chains)
            
            if self.cfg.model.auxiliary_embedding == 'localESM':
                embed = get_esm(self.cfg.data_loc.esm_data, self.cfg.data.dataset, fname, '_esm8M.pt') # [L, EMBED]
                pdb[0]['esm'] = embed
            
            self.pdb_data[i] = pdb[0]
            
    def __len__(self):
        return len(self.pdb_data)

    def __getitem__(self, index):
        """Batch retrieval fxn - do each row as its own item, for simplicity"""

        row = self.df.iloc[index]
        pdb_CANONICAL = self.pdb_data[index]

        if 'MUTS' in self.df.columns:
            mut_info = row.MUTS

            if ('ptmul' in self.cfg.data.dataset) and ('double' not in self.cfg.data.mut_types):
                if len(mut_info.split(';')) < 3: # skip double mutants if missing
                    return
            if ('ptmul' in self.cfg.data.dataset) and ('higher' not in self.cfg.data.mut_types):
                if len(mut_info.split(';')) > 2: # skip higher order mutants if missing
                    return

            # hack to run additive model on multi mutant datasets
            if self.cfg.data.get('pick', None) is not None:
                pick = self.cfg.data.get('pick', None)
                try:
                    mut_info = mut_info.split(';')[int(pick)]
                except IndexError:
                    mut_info = np.nan

        else:
            mut_info = row.MUT

        wt_list, mut_list, idx_list = [], [], []
        if mut_info is np.nan:  # to skip missing mutations for additive model
            return

        # if True: # hacky cyclic prediction setup
        #     mt1, mt2 = mut_info.split(';')[::-1] # MT2_to_DM
        #     # mt1, mt2 = mut_info.split(';') # MT1_to_DM

        #     wtAA1, pos1, mutAA1 = mt1[0], int(mt1[1:-1]) - 1, mt1[-1]
        #     wtAA2, pos2, mutAA2 = mt2[0], int(mt2[1:-1]) - 1, mt2[-1]

        #     # impute MUT1 into WT PDB data to form chimeric WT-MUT1 input
        #     real_pdb = deepcopy(pdb_CANONICAL)
        #     pdb_idx1 = self._get_pdb_idx(mt1, pdb_CANONICAL)
        #     assert real_pdb['seq'][pdb_idx1] == wtAA1
        #     pdb_idx2 = self._get_pdb_idx(mt2, pdb_CANONICAL)
        #     assert real_pdb['seq'][pdb_idx2] == wtAA2

        #     seq_keys = [k for k in real_pdb.keys() if k.startswith('seq')]
        #     if len(seq_keys) > 2:
        #         raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
        #     for sk in seq_keys:
        #         tmp = [p for p in real_pdb[sk]]
        #         tmp[pdb_idx1] = mutAA1
        #         real_pdb[sk] = ''.join(tmp)   
        #     # assert real_pdb['seq'][pos1] == wtAA1
        #     # seq = [ch for ch in real_pdb['seq']]
        #     # seq[pos1] = mutAA1
        #     # real_pdb['seq'] = ''.join(seq)

        #     # seq = [ch for ch in real_pdb['seq_chain_A']]
        #     # seq[pos1] = mutAA1
        #     # real_pdb['seq_chain_A'] = ''.join(seq)

        #     # assert real_pdb['seq'][pos1] == mutAA1
        #     # assert real_pdb['seq'][pos2] == wtAA2
            
        #     # use mt2 as "real mutation" in single-mutant model
        #     tmp_pdb = deepcopy(real_pdb)
        #     ddG = -np.inf
        #     tmp_pdb['mutation'] = Mutation([pdb_idx2], [wtAA2], [mutAA2], ddG, row.PDB[:-1])
        #     return tmp_pdb

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
            wtAA, mutAA = mut_info[0], mut_info[-1]
            ddG = float(row.DDG) * -1
            # print(f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")
            pdb_file = os.path.join(self.pdb_dir, f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")
            pdb = alt_parse_PDB(pdb_file, input_chain_list=chain, side_chains=self.side_chains)[0]
            
            pdb_idx = self._get_pdb_idx(mut_info, pdb_CANONICAL)  
            assert pdb['seq'][pdb_idx] == mutAA
            
            pdb['mutation'] = Mutation([pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])
        
        # if needed, update wt seq to match passed mutations
        seq_keys = [k for k in pdb.keys() if k.startswith('seq')]
        if len(seq_keys) > 2:
            raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
        for sk in seq_keys:
            tmp = [p for p in pdb[sk]]
            tmp[pdb_idx] = wtAA
            pdb[sk] = ''.join(tmp)   

        if self.cfg.model.auxiliary_embedding == 'localESM':
            assert 'esm' in pdb.keys() # check that each mutation has a matched ESM embedding

        tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point
        return tmp

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
    
        self.wt_names = splits[self.split]

        self.df = self.df.loc[self.df.pdb_id_corrected.isin(self.wt_names)].reset_index(drop=True)
        print('Total dataset size:', self.df.shape)

        self.side_chains = self.cfg.data.get('side_chains', False)
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.rstrip('.pdb')
            pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file, side_chains=self.side_chains)
            
            if self.cfg.model.auxiliary_embedding == 'localESM':
                embed = get_esm(self.cfg.data_loc.esm_data, self.cfg.data.dataset, wt_name, '_esm8M.pt') # [L, EMBED]
                pdb[0]['esm'] = embed
            self.pdb_data[wt_name] = pdb[0]
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):

        pdb_list = []
        row = self.df.iloc[index]
        # load PDB and correct seq as needed
        wt_name = row.pdb_id_corrected.rstrip('.pdb')
        pdb = self.pdb_data[wt_name]

        try:
            pdb_idx = row.pdb_position
            assert pdb['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
            
        except AssertionError:  # contingency for mis-alignments
            align, *rest = pairwise2.align.globalxx(row.pdb_sequence, pdb['seq'].replace("-", "X"))
            pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)

            assert pdb['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]

        ddG = float(row.ddG)
        mut = Mutation([pdb_idx], [pdb['seq'][pdb_idx]], [row.mutation], ddG, wt_name)
        
        pdb['mutation'] = mut
        
        if self.cfg.model.auxiliary_embedding == 'localESM':
            assert 'esm' in pdb.keys() # check that each mutation has a matched ESM embedding
        
        tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point

        return tmp


class MegaScaleDatasetv2(torch.utils.data.Dataset):
    """Rewritten Megascale dataset doing truly batched mutation generation (getitem returns 1 sample)"""

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split  # which split to retrieve

        if ('cdna' in self.split) or ('denovo' in self.split):
            # load ME dataset through separate process to ensure compatibility
            if self.split == 'train_cdna': # combined training set
                fname = '/home/hdieckhaus/scripts/ThermoMPNN/data/cdna_mutate_everything/cdna_train.csv'
                df = pd.read_csv(fname)
            elif self.split == 'test_cdna': # combined test set
                fname1 = '/home/hdieckhaus/scripts/ThermoMPNN/data/cdna_mutate_everything/cdna1_test.csv'
                fname2 = '/home/hdieckhaus/scripts/ThermoMPNN/data/cdna_mutate_everything/cdna2_test.csv'
                df = pd.concat([
                    pd.read_csv(fname1), 
                    pd.read_csv(fname2)
                ])
                df['WT_name'] = df['pdb_id'].str.upper() + '.pdb'
                df['ddG_ML'] = df['ddg']

            elif self.split == 'denovo':
                fname1 = '/home/hdieckhaus/scripts/ThermoMPNN/data/cdna_mutate_everything/denovo_singles.csv'
                fname2 = '/home/hdieckhaus/scripts/ThermoMPNN/data/cdna_mutate_everything/denovo_doubles.csv'
                df = pd.concat([
                    pd.read_csv(fname1), 
                    pd.read_csv(fname2)
                ])
                df['WT_name'] = df['pdb_id'] + '.pdb'
                df['ddG_ML'] = df['ddg'] * -1

                            
            # convert columns to expected names/formats
            df['aa_seq'] = df['mut_seq']
            
            df['mut_type'] = df['mut_info']

            # select singles/doubles
            singles = df.loc[~df.mut_type.str.contains(':')]
            doubles = df.loc[df.mut_type.str.contains(':')]

            # select singles/doubles
            mut_list = []
            if 'single' in self.cfg.data.mut_types:
                mut_list.append(singles)
            if 'double' in self.cfg.data.mut_types:
                mut_list.append(doubles)

            self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)
            self.df['DIRECT'] = True
            self.df['wt_orig'] = self.df['mut_type'].str[0]
            self.wt_names = self.df.WT_name.unique()

            # load PDB data
            self.side_chains = self.cfg.data.get('side_chains', False)
            self.pdb_data = {}
            for wt_name in tqdm(self.wt_names):
                wt_name = wt_name.split(".pdb")[0].replace("|",":")
                pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
                pdb = parse_PDB(pdb_file, side_chains=self.side_chains)
                self.pdb_data[wt_name] = pdb[0]

            # handle augmentation - add to df
            if 'double-aug' in self.cfg.data.mut_types:
                double_aug = self._augment_double_mutants(singles, c=1)
                print('Generated %s augmented double mutations' % str(double_aug.shape[0]))
                double_aug['DIRECT'] = False
                self.df = pd.concat([self.df, double_aug], axis=0).reset_index(drop=True)
        else:
            fname = self.cfg.data_loc.megascale_csv
            # only load rows needed to save memory
            df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq"])
            # remove unreliable data and insertion/deletion mutations
            df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)
            df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)

            mut_list = []
            if 'single' in self.cfg.data.mut_types:
                mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :].reset_index(drop=True))
            
            if 'double' in self.cfg.data.mut_types:
                tmp = df.loc[(df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :].reset_index(drop=True)
                
                # Remove hidden single mutations as these are unreliable
                mut = tmp.mut_type.values
                mut1 = [m.split(':')[0] for m in mut]
                mut2 = [m.split(':')[-1] for m in mut]
                flag = [m[0] == m[-1] for m in mut1] or [m[0] == m[-1] for m in mut2]
                flag = ~np.array(flag)
                tmp = tmp.loc[flag]
                tmp['dupe'] = tmp['WT_name'] + '_' + tmp['mut_type']
                tmp = tmp.drop_duplicates(subset=['dupe']).reset_index(drop=True)
                mut_list.append(tmp)
                
            self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)  # this includes points missing structure data
            
            # load splits produced by mmseqs clustering
            with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
                splits = pickle.load(f)

            self.wt_names = splits[self.split]

            # pre-loading wildtype structures - can avoid later file I/O for 50% of data points
            self.side_chains = self.cfg.data.get('side_chains', False)
            self.pdb_data = {}
            for wt_name in tqdm(self.wt_names):
                wt_name = wt_name.split(".pdb")[0].replace("|",":")
                pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
                pdb = parse_PDB(pdb_file, side_chains=self.side_chains)
                # load each ESM embedding in its totality
                if self.cfg.model.auxiliary_embedding == 'localESM':
                    embed = get_esm(self.cfg.data_loc.esm_data, self.cfg.data.dataset, wt_name, '_esm8M.pt') # [L, EMBED]
                    pdb[0]['esm'] = embed
                self.pdb_data[wt_name] = pdb[0]

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
                tmp = tmp.loc[tmp.WT_name.isin(self.wt_names)].reset_index(drop=True) # filter by split
                
                double_aug = self._augment_double_mutants(tmp, c=1)
                print('Generated %s augmented double mutations' % str(double_aug.shape[0]))
                double_aug['DIRECT'] = False
                self.tmp = tmp
                df_list.append(double_aug)
            
            if self.split == 'test':
                self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
            else:
                self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)
            
            epi = cfg.data.epi if 'epi' in cfg.data else False
            if epi:
                self._generate_epi_dataset()

        self._sort_dataset()
        print('Final Dataset Size: %s ' % str(self.df.shape[0])) 
        # TODO remove this later
        print('MEAN ddG:', self.df.ddG_ML.astype(float).mean())
        # self.df.to_csv('TMP.csv')
        # quit()


    def __len__(self):
        """Total sample count instead of batch count"""
        return self.df.shape[0]

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is list of protein-mutation pairs (can be different proteins)."""
        row = self.df.iloc[index]

        pdb_loc = self.cfg.data_loc.rosetta_data
        wt_name = row.WT_name.rstrip(".pdb").replace("|",":")
        chain = 'A'  # all Rocklin proteins have chain A, since they're AF2 models

        mt = row.mut_type  # only single mutations for now
        direct = row.DIRECT
        # need to retrieve file and compile mutation object
        if self.cfg.data.get('pick', None) is not None:
            pick = self.cfg.data.get('pick', None)
            mt = row.mut_type.split(':')[pick] # hacky way of running additive model on multi mutant datasets


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

                if self.side_chains:  # for now, these are stored as separate .pt files
                    pt_tag = '_sca'
                else:
                    pt_tag = ''

                pt_file = os.path.join(pdb_loc, wt_name, 'pdb_models', f'{chain}[{wt_ros}{pos_ros}{mut_ros}{pt_tag}].pt')
                # if pt exists, it's way faster to load than using the pdb parser
                if os.path.isfile(pt_file):
                    pdb = torch.load(pt_file)
                else:
                    pdb_file = os.path.join(pdb_loc, wt_name, 'pdb_models', f'{chain}[{wt_ros}{pos_ros}{mut_ros}].pdb')
                    assert os.path.isfile(pdb_file)  # check that file exists
                    pdb = parse_PDB(pdb_file, side_chains=self.side_chains)[0]
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
                pdb = parse_PDB(pdb_file, side_chains=self.side_chains)[0]
                                
        tmp_pdb = deepcopy(pdb) # this is hacky but it is needed or else it overwrites all PDBs with the last data point

        ddG = -1 * float(row.ddG_ML)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, ddG, row.WT_name)
        # return a SINGLE pdb object this way - not a batch of them
        if self.cfg.model.auxiliary_embedding == 'localESM':
            assert 'esm' in tmp_pdb.keys() # check that each mutation has a matched ESM embedding
        return tmp_pdb

    def _sort_dataset(self):
        """Sort the main df by sequence length"""
        self.df['length'] = self.df.aa_seq.str.len()
        self.df = self.df.sort_values(by='length')
        self.df.drop(columns='length')
        return

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

        # if distance weighting is enabled, load contact maps for all pdbs
        if self.cfg.data.get('weight', False):
            xyz_data = {}
            print('Using inverse distance weighting on data augmentation!')
            for un in np.unique(wtns):
                # get PDB xyz data and calculate LxL distance map
                xyz = self.pdb_data[un.removesuffix('.pdb')]['coords_chain_A']['CA_chain_A']
                xyz = torch.tensor(np.stack(xyz)) # [L, 3]
                xyz = torch.cdist(xyz, xyz)
                xyz_data[un] = xyz # [L, L]

        # if range weighting is enabled
        if self.cfg.data.range is not None:
            print('Range weighted augmentation enabled!')
            ddg_dist = new_df.loc[~new_df.mut_type.str.contains(':')]
            ddg_dist = ddg_dist.ddG_ML.values.astype(float)
            all_probs = ddg_dist * -1 # linear weighting - higher ddg means lower weight
            all_probs = all_probs - min(all_probs)
            all_probs = all_probs ** int(self.cfg.data.range)  # range coefficient is exponential scaler

        mutations = new_df.mut_type.values
        ddgs = new_df.ddG_ML.values

        oversample = self.cfg.data.oversample if 'oversample' in self.cfg.data else None
        if 'seed' in self.cfg.data:
            print('Setting data augmentation seed to %s' % str(self.cfg.data.seed))
            np.random.seed(self.cfg.data.seed)
        for p, w, m, d in tqdm(zip(positions, wtns, mutations, ddgs)):
            # pair must be in the same protein but different position, random selection
            mask = chunks[w] * ~pchunks[p]
            options = mutations[mask]
            ddgs_paired = ddgs[mask]
            if oversample == 'scale':
                probs = minmax_scale(ddgs_paired.astype(float))
                probs = probs / np.sum(probs)
            elif oversample is not None:
                # calulate Nth percentile and only keep values above this cutoff
                cutoff = np.percentile(ddgs_paired.astype(float), q=int(oversample))
                probs = (ddgs_paired.astype(float) > cutoff).astype(float)
                probs = probs / np.sum(probs)
            else:
                if self.cfg.data.weight and self.cfg.data.range is not None:
                    pos_set = positions[mask].astype(int) - 1
                    xyz_subset = xyz_data[w][int(p) - 1, :] # [L, ]
                    distances = xyz_subset[pos_set]
                    probs1 = 1. / (distances ** 2) # weighted by inverse of squared distance
                    probs1 = probs1.numpy() / np.sum(probs1.numpy())
                    probs2 = all_probs[mask]
                    probs2 = probs2 / np.sum(probs2)
                    # just avg the probs for each data point
                    probs = np.mean([probs1, probs2], axis=0)
                    # plot the distributions
                    # import matplotlib.pyplot as plt
                    # plt.scatter(probs1, probs2, s=2)
                    # plt.plot([0, .02], [0, .02], 'k--',)
                    # plt.xlabel('DIST')
                    # plt.ylabel('ddG')
                    # # plt.hist(probs1, alpha=0.5, bins=50, label='DIST')
                    # # plt.hist(probs2, alpha=0.5, bins=50, label='ddG')
                    # # plt.legend()
                    # plt.savefig('PROBS.png', dpi=300)
                    # quit()
                elif self.cfg.data.get('weight', False):
                    # weight random choice to favor nearby residues
                    pos_set = positions[mask].astype(int) - 1
                    xyz_subset = xyz_data[w][int(p) - 1, :] # [L, ]
                    distances = xyz_subset[pos_set]
                    probs = 1. / (distances ** 2) # weighted by inverse of squared distance
                    probs = probs / np.sum(probs.numpy())
                elif self.cfg.data.get('range', False):
                    probs = all_probs[mask]
                    probs = probs / np.sum(probs)
                else:
                    probs = None
                
            chosen = np.random.choice(np.arange(options.size), size=c, p=probs)

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

    def _generate_epi_dataset(self):
        """
        Convert self.df (double mutants) to epistatic dataset
        """
        print('Converting to EPI dataset')
        # split into single/double mutants
        singles = self.df.loc[~self.df.mut_type.str.contains(':')].reset_index(drop=True)
        doubles = self.df.loc[self.df.mut_type.str.count(':') == 1].reset_index(drop=True)

        doubles[['mut1', 'mut2']] = doubles.mut_type.str.split(':', n=1, expand=True)
        
        mut1 = doubles.mut1.values
        mut2 = doubles.mut2.values
        
        wtns = doubles.WT_name.values
        ddgs = doubles.ddG_ML.values.astype(float)
        bias_list = []
        additive, ddg1s, ddg2s = [], [] ,[]
        singles.ddG_ML = singles.ddG_ML.astype(float)
        
        chunks, m1chunks, m2chunks = {}, {}, {}
        sing_wtns = singles.WT_name.values
        sing_mut = singles.mut_type.values
        
        # do sweep for masks only ONCE
        for un in np.unique(wtns):
            chunk = sing_wtns == un
            chunks[un] = chunk
        
        for pos in np.unique(mut1):
            chunk = sing_mut == pos
            m1chunks[pos] = chunk
        
        for pos in np.unique(mut2):
            chunk = sing_mut == pos
            m2chunks[pos] = chunk
            
        # TODO grab additive equivalent for every double mutant
        for m1, m2, p, d in tqdm(zip(mut1, mut2, wtns, ddgs)):
            # grab each separate ddg based on PDB + mut_type matches
            mask = chunks[p] * m1chunks[m1]
            options = singles.loc[mask]
            
            mask2 = chunks[p] * m2chunks[m2]
            options2 = singles.loc[mask2]
            
            if options.shape[0] == 0:
                ddg1 = 0
            else:
                ddg1 = options.ddG_ML.values[0]
            
            if options2.shape[0] == 0:
                ddg2 = 0
            else:
                ddg2 = options2.ddG_ML.values[0]
            
            # calculate bias term from ddg1+2
            if (ddg1 == 0) or (ddg2 == 0): # if one or both single ddgs are missing, drop the data point (can't calculate epi score)
                bias = -np.inf
            else:
                bias = d - (ddg1 + ddg2)
            add = ddg1 + ddg2
    
            bias_list.append(bias)
            additive.append(add)
            ddg2s.append(ddg2)
            ddg1s.append(ddg1)
        # TODO return epistasis term in place of double mutant ddG
        doubles['ddG_actual'] = doubles.ddG_ML
        doubles.ddG_ML = bias_list
        doubles['additive'] = additive
        doubles['ddg1'] = ddg1s
        doubles['ddg2'] = ddg2s
        doubles = doubles.loc[doubles.ddG_ML != -np.inf].reset_index(drop=True)
        self.df = doubles
        return
    
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

    def _refresh_dataset(self):
        # pull out non-aug data points for re-use
        non_aug = self.df.loc[self.df['DIRECT']]
        aug = self.df.loc[~self.df['DIRECT']]
        aug = self._augment_double_mutants(df=self.tmp)
        # concat and label them as before
        print('Generated %s augmented double mutations' % str(aug.shape[0]))
        aug['DIRECT'] = False
        self.df = pd.concat([non_aug, aug], axis=0).sort_values(by='WT_name').reset_index(drop=True)
        self._sort_dataset()
        return

class SKEMPIDataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg, csv_file, pdb_loc, split='all'):
        
        self.cfg = cfg
        self.split = split
        self.df = pd.read_csv(csv_file, index_col=0)

        # drop missing values
        self.df = self.df.loc[~self.df.ddG.isna()]
        self.df = self.df.loc[~self.df.PDB_ID.isna()]

        # grab only split subset
        if self.split != 'all':
            split_file = os.path.join(os.path.dirname(csv_file), 'skempi_splits.pkl')
            with open(split_file, 'rb') as fopen:
                splits = pickle.load(fopen)
            self.df = self.df.loc[self.df.PDB_ID.isin(splits[self.split])]

        # sort by seq length for efficiency/training stability
        self.df['length'] = self.df['SEQ1'].str.len() + self.df['SEQ2'].str.len()
        self.df.sort_values(by=['length']).reset_index(drop=True)
        print('Prepped Dataset Size: %s ' % str(self.df.shape[0]))

        # pre-loading wildtype structures
        self.pdb_names = self.df.PDB_ID.unique()
        self.pdb_data = {}
        for wt_name in tqdm(self.pdb_names):
            wt_name = wt_name.removesuffix(".pdb")
            # pdb_file = os.path.join(pdb_loc, f"{wt_name}.pdb")
            # pdb = parse_PDB(pdb_file, side_chains=self.cfg.data.side_chains)[0]
            # torch.save(pdb, os.path.join(pdb_loc, f"{wt_name}.pt"))
            # print(self.pdb_names)
            # lazy quick loading - no side chain coords
            pdb = torch.load(os.path.join(pdb_loc, f"{wt_name}.pt"))
            # check that both chains were successfully loaded
            assert pdb['num_of_chains'] > 1
            self.pdb_data[wt_name] = pdb
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        row = self.df.iloc[index]
        wt_list, mut_list = [row.WT_AA], [row.MUT_AA]

        # extract key chain seq and calculate offset; add this to mut pos for final pos
        pdb = self.pdb_data[row.PDB_ID.removesuffix('.pdb')]
        offset = 0
        key_seq = f'seq_chain_{row.MUT_CHAIN}'
        for sch in [k for k in list(pdb.keys()) if k.startswith('seq_chain_')]:
            if sch == key_seq:
                break
            else:
                offset += len(pdb[sch])

        pos = offset + int(row.MUT_POS) - 1
        assert pdb['seq'][pos] == row.WT_AA
        pos_list = [pos]        
        assert row.WT_AA != row.MUT_AA

        tmp_pdb = deepcopy(pdb)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, float(row.ddG), row.PDB_ID)
        return tmp_pdb
    

class SKEMPIDoubleDataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg, csv_file, pdb_loc, split='all'):
        
        self.cfg = cfg
        self.split = split
        self.df = pd.read_csv(csv_file, index_col=0)

        # drop missing values
        self.df = self.df.loc[~self.df.ddG.isna()]
        self.df = self.df.loc[~self.df.PDB_ID.isna()]

        # grab only split subset
        if self.split != 'all':
            split_file = os.path.join(os.path.dirname(csv_file), 'skempi_splits.pkl')
            with open(split_file, 'rb') as fopen:
                splits = pickle.load(fopen)
            self.df = self.df.loc[self.df.PDB_ID.isin(splits[self.split])]

        # sort by seq length for efficiency/training stability
        self.df['length'] = self.df['SEQ1'].str.len() + self.df['SEQ2'].str.len()
        self.df.sort_values(by=['length']).reset_index(drop=True)
        print('Prepped Dataset Size: %s ' % str(self.df.shape[0]))

        # pre-loading wildtype structures
        self.pdb_names = self.df.PDB_ID.unique()
        self.pdb_data = {}
        for wt_name in tqdm(self.pdb_names):
            wt_name = wt_name.removesuffix(".pdb")
            # lazy loading from pre-processed (no SCA) PDBs
            pdb = torch.load(os.path.join(pdb_loc, f"{wt_name}.pt"))

            # full loading
            # pdb_file = os.path.join(pdb_loc, f"{wt_name}.pdb")
            # pdb = parse_PDB(pdb_file, side_chains=self.cfg.data.side_chains)[0]
            # torch.save(pdb, os.path.join(pdb_loc, f"{wt_name}.pt"))
            assert pdb['num_of_chains'] > 1
            self.pdb_data[wt_name] = pdb
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):

        def _get_offset(pdb, chain):
            offset = 0
            key_seq = f'seq_chain_{chain}'
            for sch in [k for k in list(pdb.keys()) if k.startswith('seq_chain_')]:
                if sch == key_seq:
                    break
                else:
                    offset += len(pdb[sch])
            return offset

        row = self.df.iloc[index]
        pdb = self.pdb_data[row.PDB_ID.removesuffix('.pdb')]

        mt = row['Mutation(s)_cleaned'].split(',')
        # mt = [mt[1]] # hacky additive model test
        wt_list, mut_list, pos_list = [], [], []
        for mut in mt:
            wtAA = mut[0]
            mutAA = mut[-1]
            chain = mut[1]
            pos = int(mut[2:-1]) - 1 
            offset = _get_offset(pdb, chain)
            pos = pos + offset

            wt_list.append(wtAA)
            mut_list.append(mutAA)
            pos_list.append(pos)
            assert pdb['seq'][pos] == wtAA
            assert wtAA != mutAA

        tmp_pdb = deepcopy(pdb)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, float(row.ddG), row.PDB_ID)
        return tmp_pdb
    

class S487Dataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg, csv_file, pdb_loc):
        
        self.cfg = cfg
        self.df = pd.read_csv(csv_file, index_col=None)
        self.df.columns = ['MUTINFO', 'ddG_exp', 'ddG_DLAmut']
        
        # extract PDB, mutChain, wtAA, mutAA, pos
        self.df[['PDB', 'CHAIN', 'MUT']] = self.df['MUTINFO'].str.split('_', expand=True)
        self.df['WT_AA'] = self.df['MUT'].str[0:3]
        self.df['MUT_AA'] = self.df['MUT'].str[-3:]
        self.df['POS'] = self.df['MUT'].str[3:-3].astype(int)

        self.df['WT_AA'] = self.df.apply(lambda row: seq1(row['WT_AA']), axis=1)
        self.df['MUT_AA'] = self.df.apply(lambda row: seq1(row['MUT_AA']), axis=1)

        print('Prepped Dataset Size: %s ' % str(self.df.shape[0]))

        # pre-loading wildtype structures
        self.pdb_names = self.df.PDB.unique()
        self.pdb_data = {}
        for wt_name in tqdm(self.pdb_names):
            wt_name = wt_name.removesuffix(".pdb")
            pdb_file = os.path.join(pdb_loc, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file, side_chains=self.cfg.data.side_chains)[0]
            # check that multiple chains were successfully loaded
            assert pdb['num_of_chains'] > 1
            self.pdb_data[wt_name] = pdb
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        row = self.df.iloc[index]
        wtaa = row.WT_AA
        mutaa = row.MUT_AA
        wt_list, mut_list = [wtaa], [mutaa]

        # extract key chain seq and calculate offset; add this to mut pos for final pos
        pdb = self.pdb_data[row.PDB.removesuffix('.pdb')]
        offset = 0
        key_seq = f'seq_chain_{row.CHAIN}'
        for sch in [k for k in list(pdb.keys()) if k.startswith('seq_chain_')]:
            if sch == key_seq:
                break
            else:
                offset += len(pdb[sch])

        pos = offset + int(row.POS) - 1
        assert pdb['seq'][pos] == wtaa
        pos_list = [pos]        
        assert wtaa != mutaa

        tmp_pdb = deepcopy(pdb)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, float(row.ddG_exp), row.PDB)
        return tmp_pdb
    

class PPIDataset(S487Dataset):
    def __init__(self, cfg, csv_file, pdb_loc):
        
        self.cfg = cfg
        self.df = pd.read_csv(csv_file, index_col=None, sep='\t')

        # extract PDB, mutChain, wtAA, mutAA, pos
        self.df[['PDB', 'CHAIN', 'MUT']] = self.df['mutationID'].str.split('_', expand=True)
        self.df['WT_AA'] = self.df['MUT'].str[0:3]
        self.df['MUT_AA'] = self.df['MUT'].str[-3:]
        self.df['POS'] = self.df['MUT'].str[3:-3].astype(int)
        
        self.df['WT_AA'] = self.df.apply(lambda row: seq1(row['WT_AA']), axis=1)
        self.df['MUT_AA'] = self.df.apply(lambda row: seq1(row['MUT_AA']), axis=1)

        print('Prepped Dataset Size: %s ' % str(self.df.shape[0]))

        # pre-loading wildtype structures
        self.pdb_names = self.df.PDB.unique()
        self.pdb_data = {}
        for wt_name in tqdm(self.pdb_names):
            wt_name = wt_name.removesuffix(".pdb")
            pdb_file = os.path.join(pdb_loc, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file, side_chains=self.cfg.data.side_chains)[0]
            # check that multiple chains were successfully loaded
            assert pdb['num_of_chains'] > 1
            self.pdb_data[wt_name] = pdb

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        
        row = self.df.iloc[index]
        wtaa = row.WT_AA
        mutaa = row.MUT_AA
        wt_list, mut_list = [wtaa], [mutaa]

        # extract key chain seq and calculate offset; add this to mut pos for final pos
        pdb = self.pdb_data[row.PDB.removesuffix('.pdb')]
        offset = 0
        key_seq = f'seq_chain_{row.CHAIN}'
        for sch in [k for k in list(pdb.keys()) if k.startswith('seq_chain_')]:
            if sch == key_seq:
                break
            else:
                offset += len(pdb[sch])

        pos = offset + int(row.POS) - 1
        assert pdb['seq'][pos] == wtaa
        pos_list = [pos]        
        assert wtaa != mutaa

        tmp_pdb = deepcopy(pdb)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, float(row.ddGexp), row.PDB)
        return tmp_pdb

class BinderSSMDataset(torch.utils.data.Dataset):
    """
    IPD Binder SSM Dataset
    All binders are chain A, all targets are chain B
    """

    def __init__(self, cfg, split, csv_file, pdb_loc, split_file):

        self.cfg = cfg
        self.split = split
        
        # load csv data
        self.df = pd.read_csv(csv_file, index_col=0)

        # load splits produced by mmseqs clustering
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)

        self.pdb_names = splits[self.split]

        # filter df for only data in current split
        self.df = self.df.loc[self.df.ssm_parent.isin(self.pdb_names)].reset_index(drop=True)
        
        # filter df by whether Kd is present!
        filters = self.cfg.data.get('filters', None)
    
        if 'Kd' in filters:
            print('Enabled Kd filtering')
            cols = ['parent_kd_ub', 'parent_kd_lb', 'kd_ub', 'kd_lb']
            for col in cols:
                mask = np.isfinite(self.df[col]) & (self.df[col] != 0)
                self.df = self.df.loc[mask]
        
        # filter df by region (interface etc) if desired
        if 'Interface' in filters:
            print('Enabled Interface region filtering')
            # Region options: SUPPORT, CORE, RIM (interface) ; SURFACE, INTERIOR (non-interface)
            mask = self.df['region'].isin(['SUPPORT', 'CORE', 'RIM'])
            self.df = self.df.loc[mask]
        
        # sort by seq length for efficiency/training stability
        self.df['total_length'] = self.df['binder_length'] + self.df['target_length']
        self.df.sort_values(by=['total_length']).reset_index(drop=True)
        print('Prepped Dataset Size: %s ' % str(self.df.shape[0]))

        # TODO make classifer dataset if indicated
        if self.cfg.model.classifier:
            self.df['ddG'] = make_clf_dataset(self.df['ddG'].astype(float).values)

        # pre-loading wildtype structures
        self.pdb_data = {}
        for wt_name in tqdm(self.pdb_names):
            wt_name = wt_name.removesuffix(".pdb")
            pdb_file = os.path.join(pdb_loc, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file, side_chains=self.cfg.data.side_chains)[0]
            # check that both chains were successfully loaded
            assert pdb['num_of_chains'] == 2
            self.pdb_data[wt_name] = pdb
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        """Retrieve a single mutation data point"""
        row = self.df.iloc[index]
        wt_list, mut_list, pos_list = [row.wtAA], [row.mutAA], [int(row.pos) - 1]
        # mutation is in chain A, so absolute and mutation pos should be aligned already with PDB seq
        pdb = self.pdb_data[row.ssm_parent.removesuffix('.pdb')]
        assert pdb['seq'][int(row.pos) - 1] == row.wtAA    
        assert row.wtAA != row.mutAA

        tmp_pdb = deepcopy(pdb)
        tmp_pdb['mutation'] = Mutation(pos_list, wt_list, mut_list, float(row.ddG), row.ssm_parent)
        return tmp_pdb

from math import log
from Bio import PDB, SeqUtils

def Kd_to_dG(Kd, T=298):
    R = 1.987 / 1000
    if Kd == 0:
        return 0
    dG = -1 * R * T * log(Kd)
    return dG

def get_pdb_seq(pdb_path):
    pdbparser = PDB.PDBParser(QUIET=True)
    structure = pdbparser.get_structure('chains', pdb_path)
    chains = {chain.id: SeqUtils.seq1(''.join(residue.resname for residue in chain)) for chain in structure.get_chains()}

    chains['binder_seq'] = chains.pop('A')
    chains['target_seq'] = chains.pop('B')
    return chains


def make_clf_dataset(arr):
    """Takes a numpy array of ddGs and converts it to classifier labels"""
    # stabilizing
    tmp = deepcopy(arr)
    mask = arr < -0.5
    tmp[mask] = 0
    # neutral
    mask = (arr >= -0.5) & (arr <= 0.5)
    tmp[mask] = 1
    # destabilizing
    mask = arr > 0.5
    tmp[mask] = 2 
    return tmp

class BinderSSMDatasetOmar(torch.utils.data.Dataset):
    def __init__(self, cfg, split, csv_loc, pdb_loc, split_loc):
        
        self.cfg = cfg
        self.split = split
        
        df = pd.read_csv(csv_loc, sep=' ')
        
        df = df.loc[~df['sketch_kd']]
        df = df.loc[~df['is_native']]
        lb = df['lowest_conc'] / 10
        ub = df['highest_conc'] * 1e8
        df['kd_center'] = np.sqrt(df['kd_lb'].clip(lb, ub) * df['kd_ub'].clip(lb, ub))
        df['parent_kd_center'] = np.sqrt(df['parent_kd_lb'].clip(lb, ub) * df['parent_kd_ub'].clip(lb, ub))
        
        df['dg_center'] = df['kd_center'].apply(Kd_to_dG)
        df['parent_dg_center'] = df['parent_kd_center'].apply(Kd_to_dG)
        df['ddg'] = df['parent_dg_center'] - df['dg_center']

        # do Kd filtering if enabled
        if self.cfg.data.get('filter', False):
            print('Enabled Kd filtering')
            cols = ['parent_kd_ub', 'parent_kd_lb', 'kd_ub', 'kd_lb']
            for col in cols:
                mask = np.isfinite(df[col]) & (df[col] != 0)
                df = df.loc[mask]

        seqs_df = df.drop_duplicates(subset=['ssm_parent']).copy()
        self.pdb_dir = pdb_loc
        seqs_df['ssm_parent_path'] = pdb_loc + '/' + df['ssm_parent'] + '.pdb'

        chains = seqs_df['ssm_parent_path'].apply(get_pdb_seq)
        chains_df = chains.apply(pd.Series)
        
        seqs_df = seqs_df.join(chains_df)
        df = df.merge(seqs_df[['ssm_parent', 'target', 'binder_seq', 'target_seq']], on=['ssm_parent', 'target'], how='left')
        df[['description', 'pos', 'mut_to']] = df['description'].str.split('__', expand=True)
        
        self.df = df
        with open(split_loc, 'rb') as fh:
            splits = pickle.load(fh)  
        
        self.df = self.df.loc[self.df['ssm_parent'].isin(splits[split])]
        print('Final Dataset Size: ', self.df.shape[0])
        self.df['length'] = self.df.binder_seq.str.len() + self.df.target_seq.str.len()
        self.df = self.df.sort_values(by=['length']).reset_index(drop=True)
        
        self.pdb_data = {}
        # for wt_name in tqdm(splits[split]):
        for wt_name in tqdm(self.df.ssm_parent.unique()):
            wt_name = wt_name.removesuffix('.pdb')
            pdb_file = os.path.join(pdb_loc, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file)[0]
            # pt_file = os.path.join(pdb_loc, f'{wt_name}.pt')
            # torch.save(pdb, pt_file)
            self.pdb_data[wt_name] = pdb
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        pdb = self.pdb_data[row.ssm_parent]
        pdb_idx = int(row.pos) - 1
        mutation = Mutation(position=[pdb_idx], wildtype=[pdb['seq'][pdb_idx]], mutation=[row.mut_to], ddG=float(row.ddg), pdb=row.ssm_parent)
        tmp = deepcopy(pdb)
        tmp['mutation'] = mutation
        return tmp  
        

class ProteinGymDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        df = pd.read_csv(csv_fname)
        self.df = df

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()
                 
        self.pdb_data = {}
        self.side_chains = self.cfg.data.get('side_chains', False)
        # parse all PDBs first - treat each row as its own PDB
        pdbs = self.df.PDB.unique()
        for p in tqdm(pdbs):
            fname = p[:-1]
            pdb_file = os.path.join(self.pdb_dir, f"{fname}.pdb")
            chain = [p[-1]]
            pdb = alt_parse_PDB(pdb_file, input_chain_list=chain, side_chains=self.side_chains)
            self.pdb_data[p] = pdb[0]
            
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """Batch retrieval fxn - do each row as its own item, for simplicity"""

        row = self.df.iloc[index]
        pdb_CANONICAL = self.pdb_data[row.PDB]

        if 'MUTS' in self.df.columns:
            mut_info = row.MUTS

            # hack to run additive model on multi mutant datasets
            if self.cfg.data.get('pick', None) is not None:
                pick = self.cfg.data.get('pick', None)
                try:
                    mut_info = mut_info.split(';')[int(pick)]
                except IndexError:
                    mut_info = np.nan

        else:
            mut_info = row.MUT

        wt_list, mut_list, idx_list = [], [], []
        if mut_info is np.nan:  # to skip missing mutations for additive model
            return

        for mt in mut_info.split(';'):  # handle multiple mutations like for megascale
            
            wtAA, mutAA = mt[0], mt[-1]
            ddG = float(row.DDG) * -1
            
            pdb = deepcopy(pdb_CANONICAL)      
            pdb_idx = int(mt[1:-1]) - 1
            assert pdb['seq'][pdb_idx] == wtAA
            
            wt_list.append(wtAA)
            mut_list.append(mutAA)
            idx_list.append(pdb_idx)
            pdb['mutation'] = Mutation(idx_list, wt_list, mut_list, ddG, row.PDB[:-1])

        tmp = deepcopy(pdb)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point
        return tmp



def get_esm(location, dataset, pdb, suffix):
    """Return a specific ESM embedding from disk"""
    location = os.path.join(location, dataset)
    location = os.path.join(location, f"{pdb}_esm8M.pt")
    return torch.load(location)

def prebatch_dataset(dataset, workers=1):
    """Runs pre-batching for large (augmented) datasets"""
    from torch.utils.data import DataLoader
    print('Starting Prebatching for dataset...')
    print('Number of workers:', workers)

    loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=0, batch_size=1)
    
    for batch in tqdm(loader):
        pass
    
    return

if __name__ == "__main__":
    # testing functionality
    from omegaconf import OmegaConf
    import sys
    from train_thermompnn import parse_cfg

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))
    cfg = parse_cfg(cfg)

    seed = 111 # SEEDS: 111, 222, and 333 for ESM trials
    cfg.data.seed = seed
    # prefetch augmented dataset; save to disk for embedding use
    for split in ['train_ptmul']: #, 'val', 'test']:
        ds = MegaScaleDatasetv2(cfg, split=split)
        print(ds.df)
        ds.df.to_csv(f'Prefetch_Mega_{split}_{seed}.csv')
    
    # PTMUL df retrieval (w/aligned positions)
    # pdb_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/pdbs')
    # csv_loc = os.path.join(cfg.data_loc.misc_data, 'protddg-bench-master/PTMUL/ptmul-5fold-mutateeverything_FINAL.csv')
    # ds = ddgBenchDatasetv2(cfg=cfg, csv_fname=csv_loc, pdb_dir=pdb_loc)
    # pos1_list, pos2_list = [], []
    # seq_list = []
    # for batch in tqdm(ds):
    #     if batch is not None:
    #         plist = batch['mutation'].position
    #         seq_list.append(batch['seq'])
    #         pos1_list.append(plist[0])
    #         pos2_list.append(plist[1])
    #     else:
    #         pos1_list.append(None)
    #         pos2_list.append(None)
    #         seq_list.append(None)
    
    # ds.df['Pos1_Aligned'] = pos1_list
    # ds.df['Pos2_Aligned'] = pos2_list
    # ds.df['Seq_Aligned'] = seq_list
    # ds.df = ds.df.loc[ds.df['NMUT'] == 2]
    # print(ds.df.head)
    # ds.df['Pos1_Aligned'] = ds.df['Pos1_Aligned'].astype(int)
    # ds.df['Pos2_Aligned'] = ds.df['Pos2_Aligned'].astype(int)
    # ds.df.to_csv('PTMUL-aligned.csv')
    
    # csvf = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/SKEMPIv2/SKEMPI_v2_single.csv'
    # pdbd = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/SKEMPIv2/PDBs'
    # dataset = SKEMPIDataset(cfg, csv_file=csvf, pdb_loc=pdbd)
    
    # prebatch_dataset(dataset=dataset, workers=cfg.training.num_workers)
    # prebatch_dataset(dataset=BinderSSMDataset(cfg, split, csv_file=csvf, pdb_loc=pdbd, split_file=splitf),
                    #  workers=cfg.training.num_workers)
