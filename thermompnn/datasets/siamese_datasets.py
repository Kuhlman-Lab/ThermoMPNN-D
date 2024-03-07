import torch
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from tqdm import tqdm
from copy import deepcopy

from thermompnn.protein_mpnn_utils import alt_parse_PDB, parse_PDB
from thermompnn.datasets.dataset_utils import Mutation, seq1_index_to_seq2_index
from thermompnn.datasets.v2_datasets import tied_featurize_mut


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


def adj_sequence(pdb, wt, mut, idx):
    """
    Updates pdb sequence from wt to mut at position idx (e.g., 53)
    """
    seq_keys = [k for k in pdb.keys() if k.startswith('seq')]
    if len(seq_keys) > 2:
        raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
    for sk in seq_keys:
        tmp = [p for p in pdb[sk]]
        assert tmp[idx] == wt
        tmp[idx] = mut
        pdb[sk] = ''.join(tmp)
    return pdb

class SsymDatasetSiamese(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname, flip=False, mut_struct=False):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        self.rev = flip  # "reverse" mutation testing
        self.mut_struct = mut_struct

        df = pd.read_csv(csv_fname)
        self.df = df

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()
                 
        self.pdb_data = {}
        
        # split into dir and inv dfs
        self.dir = self.df.loc[self.df['DIR/INV '] == 'DIR'].reset_index(drop=True)
        self.inv = self.df.loc[self.df['DIR/INV '] == 'INV'].reset_index(drop=True)

        assert self.dir.shape == self.inv.shape
        # parse all PDBs first - treat each row as its own PDB
        for i, row in self.dir.iterrows():
            self.pdb_data[i] = {
                'DIR': None,
                'INV': None
            }

            # dir PDB
            row = self.dir.iloc[i]
            fname = row.PDB[:-1]
            pdb_file = os.path.join(self.pdb_dir, f"{fname}.pdb")
            chain = [row.PDB[-1]]
            pdb = alt_parse_PDB(pdb_file, chain)[0]
            self.pdb_data[i]['DIR'] = pdb

            # inv PDB
            row = self.inv.iloc[i]
            fname = row.PDB[:-1]
            pdb_file = os.path.join(self.pdb_dir, f"{fname}.pdb")
            chain = [row.PDB[-1]]
            pdb = alt_parse_PDB(pdb_file, chain)[0]
            self.pdb_data[i]['INV'] = pdb

    def __len__(self):
        return len(self.pdb_data)

    def __getitem__(self, index):
        """Batch retrieval fxn - do each row as its own batch, for simplicity"""

        # make direct mutation
        row = self.dir.iloc[index]
        pdb_CANONICAL = self.pdb_data[index]['DIR']
        mut_info = row.MUT
    
        wtAA, mutAA = mut_info[0], mut_info[-1]
        pdb_idx = self._get_pdb_idx(mut_info, pdb_CANONICAL)
        wt_pdb_idx = pdb_idx
        ddG = float(row.DDG) * -1

        pdb = deepcopy(pdb_CANONICAL)
        assert pdb['seq'][pdb_idx] == wtAA
        pdb['mutation'] = Mutation([pdb_idx], [wtAA], [mutAA], ddG, row.PDB[:-1])    

        # make inverse mutation
        row = self.inv.iloc[index]
        pdb_CANONICAL = self.pdb_data[index]['INV']
        mut_info = row.MUT
    
        wtAA, mutAA = mut_info[0], mut_info[-1]
        pdb_idx = self._get_pdb_idx(mut_info, pdb_CANONICAL)
        mut_pdb_idx = pdb_idx
        ddG = float(row.DDG) * -1

        mut_pdb = deepcopy(pdb_CANONICAL)
        assert mut_pdb['seq'][pdb_idx] == wtAA
        mut_pdb['mutation'] = Mutation([pdb_idx], [wtAA], [mutAA], ddG, row.PDB[:-1])    

        if not self.mut_struct and self.rev:  # impute the DIRECT ssym structure
            pdb = deepcopy(mut_pdb)
            pdb = adj_sequence(pdb, wtAA, mutAA, mut_pdb_idx)
            pdb['mutation'] = Mutation([mut_pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])

        elif not self.mut_struct and not self.rev:  # input the INVERSE ssym structure
            mut_pdb = deepcopy(pdb)
            mut_pdb = adj_sequence(mut_pdb, mutAA, wtAA, wt_pdb_idx)
            mut_pdb['mutation'] = Mutation([wt_pdb_idx], [wtAA], [mutAA], ddG, row.PDB[:-1])

        features = tied_featurize_mut([pdb], 'cpu')
        mut_features = tied_featurize_mut([mut_pdb], 'cpu')

        if self.rev:
            return mut_features, features
        else:
            return features, mut_features
    

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
            # if gaps are present, add these to idx (+10 to get any around the mutation site, kinda a hack)
            if 'S669' in self.pdb_dir:
                gaps = [g for g in pdb['seq'] if g == '-']
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


class ddgBenchDatasetSiamese(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname, fwd='xtal', back='rosetta'):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        self.fwd = fwd
        self.back = back

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
        """Batch retrieval fxn - do each row as its own batch, for simplicity"""

        feature_list = []
        for grab in (self.fwd, self.back):
            row = self.df.iloc[index]
            pdb_CANONICAL = self.pdb_data[index]
            mut_info = row.MUT

            # make direct mutation
            wtAA, mutAA = mut_info[0], mut_info[-1]
            pdb_idx = self._get_pdb_idx(mut_info, pdb_CANONICAL)
            ddG = float(row.DDG) * -1

            if grab.startswith('xtal'):  # use xtal structure without modification
                pdb = deepcopy(pdb_CANONICAL)
                assert pdb['seq'][pdb_idx] == wtAA
                if not grab.endswith('flipped'):
                    # print('Using normal xtal')
                    pdb['mutation'] = Mutation([pdb_idx], [wtAA], [mutAA], ddG, row.PDB[:-1])    
                else: # if flipped, get synthetic reverse mutation
                    pdb = adj_sequence(pdb, wtAA, mutAA, pdb_idx)
                    pdb['mutation'] = Mutation([pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])
            
            elif grab.startswith('rosetta'):
                fname = row.PDB[:-1]
                chain = row.PDB[-1]
                old_idx = int(mut_info[1:-1])
                pdb_file = os.path.join(self.pdb_dir, f"{fname}{chain}_{wtAA}{old_idx}{mutAA}_relaxed.pdb")

                tmp = alt_parse_PDB(pdb_file, chain)[0]
                pdb = deepcopy(tmp)
                if not grab.endswith('flipped'):  # use Rosetta structure as-is
                    pdb['mutation'] = Mutation([pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])
                else:
                    seq_keys = [k for k in pdb.keys() if k.startswith('seq')]
                    if len(seq_keys) > 2:
                        raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
                    for sk in seq_keys:
                        tmp = [p for p in pdb[sk]]
                        tmp[pdb_idx] = wtAA
                        pdb[sk] = ''.join(tmp)
                    pdb['mutation'] = Mutation([pdb_idx], [wtAA], [mutAA], ddG, row.PDB[:-1])
            else:
                pass

            features = tied_featurize_mut([pdb], 'cpu')
            feature_list.append(features)

        return (feature_list[0], feature_list[1])
    
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
            # if gaps are present, add these to idx (+10 to get any around the mutation site, kinda a hack)
            if 'S669' in self.pdb_dir:
                gaps = [g for g in pdb['seq'] if g == '-']
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


class FireProtDatasetSiamese(torch.utils.data.Dataset):

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
        self.cfg.training.batch_size = 100
        self.clusters = batchify(self.df.pdb_sequence.str.len().values, self.cfg.training.batch_size)
        print('Generated %s batches of size %s for %s split' % (str(len(self.clusters)), str(self.cfg.training.batch_size), self.split))

        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.rstrip('.pdb')
            pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file)
            self.pdb_data[wt_name] = pdb[0]
        
    def _batchify(self):
        # generate clusters of df idx values for batching
        # adapted from proteinmpnn.utils.StructureLoader
        # grab length of each seq
        lengths = self.df.pdb_sequence.str.len().values

        print('Batch size:', self.cfg.training.batch_size)
        # argsort returns indexes, not sorted list
        sorted_ix = np.argsort(lengths)
        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = lengths[ix]
            if size * (len(batch) + 1) <= self.cfg.training.batch_size:  # make sure new size (B x L_max) is under batch_size
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(np.array(batch))
                batch, batch_max = [ix], size
            
        if len(batch) > 0:
            clusters.append(np.array(batch))
        # self.clusters is a list of (lists of indexes which each make up a batch with length <= self.batch_size)
        self.clusters = clusters
        return

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, index):

        batch_idx = self.clusters[index]
        pdb_list, mut_pdb_list = [], []
        for i, row in self.df.iloc[batch_idx].iterrows():
            # load PDB and correct seq as needed
            wt_name = row.pdb_id_corrected.rstrip('.pdb')
            pdb_CANONICAL = self.pdb_data[wt_name]
            pdb = deepcopy(pdb_CANONICAL)  # this is hacky but it is needed or else it overwrites all PDBs with the last data point

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
            pdb_list.append(pdb)

            # make seq flipped
            tmp_pdb = deepcopy(pdb_CANONICAL)
            pos = pdb_idx

            # making synthetic MUT PDB object                
            seq_keys = [k for k in tmp_pdb.keys() if k.startswith('seq')]
            if len(seq_keys) > 2:
                raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
            for sk in seq_keys:
                tmp = [p for p in tmp_pdb[sk]]
                tmp[pos] = row.mutation
                tmp_pdb[sk] = ''.join(tmp)
            # check that both seqs got changed
            assert tmp_pdb[seq_keys[0]] == tmp_pdb[seq_keys[1]]
            assert tmp_pdb['seq'][pos] == row.mutation

            mut = Mutation([pdb_idx], [row.mutation], [row.wild_type], ddG * -1, wt_name)
            tmp_pdb['mutation'] = mut
            mut_pdb_list.append(tmp_pdb)

        features = tied_featurize_mut(pdb_list, 'cpu')
        mut_features = tied_featurize_mut(mut_pdb_list, 'cpu')
        return features, mut_features


class MegaScaleDatasetSiamese(torch.utils.data.Dataset):
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

        self.df = pd.concat(df_list, axis=0).sort_values(by='WT_name').reset_index(drop=True)
        print('Final Dataset Size: %s ' % str(self.df.shape[0]))

        # generate batches (lists of df idx for pulling data and matching with PDBs)
        print('Batch size:', self.cfg.training.batch_size)
        self._batchify()
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
        wt_pdb_list, mut_pdb_list = [], []

        for i, row in self.df.iloc[batch_idx].iterrows():
            # load PDB and correct seq as needed
            pdb_CANONICAL = self.pdb_data[row.WT_name.strip('.pdb')]
                        
            mut_types = row.mut_type.split(':')
            assert len(mut_types) == 1

            for mt in mut_types:
                # making WT PDB object
                pdb = deepcopy(pdb_CANONICAL)  # this is CRUCIAL - do not remove; otherwise any changes will affect ALL pdbs
                wt, pos, mut = mt[0], int(mt[1:-1]) - 1, mt[-1]
                ddG = -1 * float(row.ddG_ML)
                # print(ddG)
                assert pdb['seq'][pos] == wt
                pdb['mutation'] = Mutation([pos], [wt], [mut], ddG, row.WT_name)
                tmp_pdb = deepcopy(pdb)  # this is CRUCIAL - do not remove; otherwise any changes will affect ALL pdbs
                wt_pdb_list.append(pdb)
                
                # making synthetic MUT PDB object                
                seq_keys = [k for k in tmp_pdb.keys() if k.startswith('seq')]
                if len(seq_keys) > 2:
                    raise ValueError("Maximum of 2 seq fields expected in PDB, %s seq fields found instead" % str(len(seq_keys)))
                for sk in seq_keys:
                    tmp = [p for p in tmp_pdb[sk]]
                    tmp[pos] = mut
                    tmp_pdb[sk] = ''.join(tmp)
                # check that both seqs got changed
                assert tmp_pdb[seq_keys[0]] == tmp_pdb[seq_keys[1]]
                assert tmp_pdb['seq'][pos] == mut
                tmp_pdb['mutation'] = Mutation([pos], [mut], [wt], ddG * -1, row.WT_name)
                mut_pdb_list.append(tmp_pdb)
        
        # putting tied_featurize here means the CPU, not the GPU, handles it, and it is parallelized to each DataLoader
        wt_features = tied_featurize_mut(wt_pdb_list, 'cpu')
        mut_features = tied_featurize_mut(mut_pdb_list, 'cpu')

        return wt_features, mut_features

    def _batchify(self):
        # generate clusters of df idx values for batching
        # adapted from proteinmpnn.utils.StructureLoader
        # grab length of each seq
        lengths = self.df.aa_seq.str.len().values

        # argsort returns indexes, not sorted list
        sorted_ix = np.argsort(lengths)
        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = lengths[ix]
            if size * (len(batch) + 1) <= self.cfg.training.batch_size:  # make sure new size (B x L_max) is under batch_size
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(np.array(batch))
                batch, batch_max = [ix], size
            
        if len(batch) > 0:
            clusters.append(np.array(batch))
        # self.clusters is a list of (lists of indexes which each make up a batch with length <= self.batch_size)
        self.clusters = clusters
        return


class MegaScaleDatasetSiameseAug(torch.utils.data.Dataset):
    """Rewritten Megascale dataset doing batched mutation generation
    Siamese network construction"""
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
            
    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch includes list of WT PDBs, list of Mutant PDBs, and list of mutations"""
        
        batch_idx = self.clusters[index]
        wt_pdb_list, mut_pdb_list = [], []
        
        for i, row in self.df.iloc[batch_idx].iterrows():

            pdb_loc = '/work/users/d/i/dieckhau/rocklin_data/FINAL_results/'
            # TODO henry load PDB data on-the-fly from Rosetta simulations
            wt_name = row.WT_name.rstrip(".pdb").replace("|",":")
            chain = 'A'  # all Rocklin proteins have chain A, since they're monomer design models
            mt = row.mut_type  # only single mutations for now
            wt = mt[0]
            mut = mt[-1]
            pos = int(mt[1:-1])  # rosetta numbering starts at 1, NOT 0

            # grab pre-loaded WT protein
            pdb = self.pdb_data[row.WT_name.strip('.pdb')]
            pdb_wt = deepcopy(pdb) # this is hacky but it is needed or else it overwrites all PDBs with the last data point

            # grab mutant structure for every data point
            pdb_file = os.path.join(pdb_loc, 
                                    wt_name, 
                                    'pdb_models', 
                                    f'{chain}[{wt}{pos}{mut}].pdb')
            assert os.path.isfile(pdb_file)  # check that file exists
            # return
            tmp_mut = parse_PDB(pdb_file)[0]
            pdb_mut = deepcopy(tmp_mut)
                                
            pos -= 1  # adjust for sequence indexing (starts at 0)

            # check for correct residues
            assert pdb_wt['seq'][pos] == wt and pdb_mut['seq'][pos] == mut

            ddG = -1 * float(row.ddG_ML)
            pdb_wt['mutation'] = Mutation([pos], [wt], [mut], ddG, row.WT_name)
            pdb_mut['mutation'] = Mutation([pos], [mut], [wt], ddG * -1, row.WT_name)
            wt_pdb_list.append(pdb_wt)
            mut_pdb_list.append(pdb_mut)


        # # putting tied_featurize here means the CPU, not the GPU, handles it, and it is parallelized to each DataLoader
        wt_features = tied_featurize_mut(wt_pdb_list, 'cpu')
        mut_features = tied_featurize_mut(mut_pdb_list, 'cpu')

        # # save as .pt file for later loading
        fpath = os.path.join('data/mega_scale/batched_TR_Siamese/%s' % self.split, f'batch_{index}_WT.pt')
        torch.save(wt_features, fpath)
        
        fpath = os.path.join('data/mega_scale/batched_TR_Siamese/%s' % self.split, f'batch_{index}_MUT.pt')
        torch.save(mut_features, fpath)
            
        print('Saved batch %s' % str(index))
        return

class MegaScaleDatasetSiamesePt(torch.utils.data.Dataset):
    """Rewritten Megascale dataset loading individual .pt files as batches"""

    def __init__(self, cfg, split):
        print('Retrieving Megascale-Siamese-Pt dataset for split %s' % split)
        self.cfg = cfg
        self.split = split  # which split to retrieve
        
        self.pt_loc = os.path.join('data/mega_scale/batched_TR_Siamese', self.split)

        self.batch_files = sorted(os.listdir(self.pt_loc))
        self.batch_files = [sb for sb in self.batch_files if sb.endswith('.pt')]
        
        self.mut_batch_files = [sb for sb in self.batch_files if 'MUT' in sb]
        self.wt_batch_files = [sb for sb in self.batch_files if 'WT' in sb]
        del self.batch_files
    
    def __len__(self):
        return len(self.wt_batch_files)

    def __getitem__(self, index):
        """
        Batch retrieval fxn - each batch is pre-packed into a .pt file
        Separate batch files for WT and MUT feature batches
        """
        
        current_batch = f'batch_{index}_WT.pt'
        current_batch = os.path.join(self.pt_loc, current_batch)
        wt_features = torch.load(current_batch)
        
        current_batch = f'batch_{index}_MUT.pt'
        current_batch = os.path.join(self.pt_loc, current_batch)
        mut_features = torch.load(current_batch)
        
        return wt_features, mut_features


if __name__ == "__main__":
    # testing functionality
    from omegaconf import OmegaConf
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cfg = OmegaConf.merge(OmegaConf.load('../../DEBUG.yaml'), OmegaConf.load('../../local.yaml'))   
    
    # ds = MegaScaleDatasetSiamese(cfg, 'test')
    # ds = MegaScaleDatasetv2Aug(cfg, "val")
    ds = FireProtDatasetSiamese(cfg, 'test')
    # ds = ddgBenchDatasetSiamese(cfg, '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/protddg-bench-master/P53/pdbs', '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/protddg-bench-master/P53/p53_clean.csv')

    print('Starting dataset iteration')
    for batch in tqdm(ds):
        pass