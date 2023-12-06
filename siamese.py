import torch
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import itertools
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, R2Score, SpearmanCorrCoef, PearsonCorrCoef
from torchmetrics import Metric
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import math

from datasets import parse_pdb_cached, Mutation, seq1_index_to_seq2_index
from protein_mpnn_utils import alt_parse_PDB, parse_PDB
from transfer_model import ALPHABET, get_protein_mpnn, LightAttention
from batched_thermompnn import tied_featurize_mut, get_metrics_new
from Bio import pairwise2


def adj_sequence(pdb, wt, mut, idx):
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
        self._batchify()
        print('Generated %s batches of size %s for %s split' % (str(len(self.clusters)), str(self.cfg.training.batch_size), self.split))

        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.rstrip('.pdb')
            pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{wt_name}.pdb")
            pdb = parse_pdb_cached(self.cfg, pdb_file)
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
            pdb = parse_pdb_cached(self.cfg, pdb_file)
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
        self._batchify()
        print('Generated %s batches of size %s for %s split' % (str(len(self.clusters)), str(self.cfg.training.batch_size), self.split))

        # pre-loading wildtype structures - can avoid file I/O for 50% of data points
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.split(".pdb")[0].replace("|",":")
            pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
            pdb = parse_pdb_cached(self.cfg, pdb_file)
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

    def _batchify(self):
        # generate clusters of df idx values for batching
        # adapted from proteinmpnn.utils.StructureLoader
        # grab length of each seq
        lengths = self.df.aa_seq.str.len().values

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
        # self.clusters is a list of (lists of indexes, which each make up a batch with length <= self.batch_size)
        self.clusters = clusters
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

    
def custom_log_cosh_loss(D, y ,S):
    def _log_cosh(D, y, S):
        # logcosh minimum is at 0, so this minimizes true/pred difference
        # S is the avg ddg which should also be minimized down to 0
        return torch.log(torch.cosh(D - y)) + torch.abs(S)
    return torch.mean(_log_cosh(D, y, S))

class CustomLogCoshLoss(torch.nn.Module):
    """Regression loss fxn used in ACDC-NN (NOT standard log-cosh loss)"""
    def __init__(self):
        super().__init__()

    def forward(self, D, y, S):
        # D is target_ddg, y is true_ddg, S is avg_ddg
        return custom_log_cosh_loss(D, y, S)


class CustomLogCoshMetric(Metric):
    def __init__(self):
        super().__init__()
        # define tracked variables
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, D, y, S):
        # add next metric batch to total and count number of elements
        assert D.shape == y.shape == S.shape
        
        self.total_loss += torch.sum(custom_log_cosh_loss(D, y, S))
        self.total += 1

    def compute(self):
        # get avg of metric
        return self.total_loss.float() / self.total
    
def log_cosh_only(D, y ,S):
    def _log_cosh(D, y, S):
        # logcosh minimum is at 0, so this minimizes true/pred difference
        # S is the avg ddg which should also be minimized down to 0
        return torch.log(torch.cosh(D - y))
    return torch.mean(_log_cosh(D, y, S))

class LogCoshOnly(Metric):
    """Only first loss term (for metric tracking)"""
    def __init__(self):
        super().__init__()
        # define tracked variables
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, D, y, S):
        # add next metric batch to total and count number of elements
        assert D.shape == y.shape == S.shape
        
        self.total_loss += torch.sum(log_cosh_only(D, y, S))
        self.total += 1

    def compute(self):
        # get avg of metric
        return self.total_loss.float() / self.total

def abs_loss_only(D, y ,S):
    def _log_cosh(D, y, S):
        # logcosh minimum is at 0, so this minimizes true/pred difference
        # S is the avg ddg which should also be minimized down to 0
        return torch.abs(S)
    return torch.mean(_log_cosh(D, y, S))

class AbsLossOnly(Metric):
    """Only first loss term (for metric tracking)"""
    def __init__(self):
        super().__init__()
        # define tracked variables
        self.add_state("total_loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, D, y, S):
        # add next metric batch to total and count number of elements
        assert D.shape == y.shape == S.shape
        
        self.total_loss += torch.sum(abs_loss_only(D, y, S))
        self.total += 1

    def compute(self):
        # get avg of metric
        return self.total_loss.float() / self.total
    

class TransferModelSiamese(nn.Module):
    """Rewritten TransferModel class using Siamese training and Batched datasets"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dims = list(cfg.model.hidden_dims)
        print('Siamese Model Enabled!')

        self.num_final_layers = cfg.model.num_final_layers

        if 'decoding_order' not in self.cfg:
            self.cfg.decoding_order = 'left-to-right'
        
        self.prot_mpnn = get_protein_mpnn(cfg)
        
        EMBED_DIM = 128 * 2
        HIDDEN_DIM = 128
        VOCAB_DIM = 1

        # modify input size if multi mutations used
        hid_sizes = [(HIDDEN_DIM*self.num_final_layers + EMBED_DIM)]
        hid_sizes += self.hidden_dims
        hid_sizes += [ VOCAB_DIM ]

        print('MLP HIDDEN SIZES:', hid_sizes)
        
        self.lightattn = cfg.model.lightattn if 'lightattn' in cfg.model else False
        
        if self.lightattn:
            print('Enabled LightAttention')
            self.light_attention = LightAttention(embeddings_dim=(HIDDEN_DIM*self.num_final_layers + EMBED_DIM))

        self.ddg_out = nn.Sequential()

        for sz1, sz2 in zip(hid_sizes, hid_sizes[1:]):
            self.ddg_out.append(nn.ReLU())
            self.ddg_out.append(nn.Linear(sz1, sz2))

    def forward(self, wt_features, mut_features):
        """Vectorized fwd function for arbitrary batches of mutations"""
        
        features = [(wt_features, mut_features), (mut_features, wt_features)]
        ddg_both = []
        
        for feature_pair in features:
            # feat1 is the "main" set, feat2 gets used for mutant residue embedding retrieval
            feat1, feat2 = feature_pair
            # grab current features
            X, S, mask, lengths, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = feat1
            # grab mutant sequence for embedding generation
            S2 = feat2[1]
            
            # getting ProteinMPNN structure embeddings
            all_mpnn_hid, seq_embed, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
        
            if self.num_final_layers > 0:
                all_mpnn_hid = torch.cat(all_mpnn_hid[:self.num_final_layers], -1)
                # grab mutant sequence embedding
                if S2.shape != S.shape:
                    S2 = self._fix_mutant_seq(S, S2, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)
                mutant_seq_embed = self.prot_mpnn.W_s(S2)
                mpnn_embed = torch.cat([all_mpnn_hid, seq_embed, mutant_seq_embed], -1)
            
            # vectorized indexing of the embeddings (this is very ugly but the best I can do for now)
            # unsqueeze gets mut_pos to shape (batch, 1, 1), then this is copied with expand to be shape (batch, 1, embed_dim) for gather
            mpnn_embed = torch.gather(mpnn_embed, 1, mut_positions.unsqueeze(-1).expand(mut_positions.size(0), mut_positions.size(1), mpnn_embed.size(2)))
            mpnn_embed = torch.squeeze(mpnn_embed, 1) # final shape: (batch, embed_dim)
            
            # pass through lightattn
            if self.lightattn:
                mpnn_embed = torch.unsqueeze(mpnn_embed, -1)  # shape for LA input: (batch, embed_dim, seq_length=1)
                mpnn_embed = self.light_attention(mpnn_embed, mask)  # shape for LA output: (batch, embed_dim)

            ddg = self.ddg_out(mpnn_embed)  # shape: (batch, 1)
            ddg_both.append(ddg)

        # first pred is the direct mutation, second is the reverse mutations            
        ddg1, ddg2 = ddg_both
        return ddg1, ddg2

    def _fix_mutant_seq(self, S, S2, wtAA, mutAA, positions):
        """
        If mutant seq isn't a match, make it synthetically
        Only works for batch size = 1 for now
        """
        new_S2 = torch.clone(S)
        assert new_S2[0, positions[0, 0]] == wtAA[0, 0]  # check you've got the right spot
        new_S2[0, positions[0, 0]] = mutAA[0, 0]  # overwrite AA embedding manually
        return new_S2

class TransferModelSiamesePL(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, cfg):
        super().__init__()
        self.model = TransferModelSiamese(cfg)

        self.cfg = cfg
        self.learn_rate = cfg.training.learn_rate
        self.mpnn_learn_rate = cfg.training.mpnn_learn_rate if 'mpnn_learn_rate' in cfg.training else None
        self.lr_schedule = cfg.training.lr_schedule if 'lr_schedule' in cfg.training else False
        self.avg_multiplier = float(self.cfg.training.avg_multiplier) if 'avg_multiplier' in self.cfg.training else 1.0
        print('AVG MULTIPLIER:', self.avg_multiplier)
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.loss = cfg.training.loss_fxn if 'loss_fxn' in cfg.training else 'mse'
        if self.loss != 'mse':
            print('Using custom COSH loss')
            self.log_cosh = CustomLogCoshLoss()

        # set up metrics dictionary
        self.extra_metrics = nn.ModuleDict()
        self.metrics = nn.ModuleDict()
        for split in ("train_metrics", "val_metrics"):
            self.metrics[split] = nn.ModuleDict()
            out = "ddG"
            self.metrics[split][out] = nn.ModuleDict()
            for name, metric in get_metrics_new().items():
                self.metrics[split][out][name] = metric
            
            # TODO initialize custom metrics
            # if self.loss != 'mse':
            self.extra_metrics[split] = nn.ModuleDict()
            self.extra_metrics[split][out] = nn.ModuleDict()
            self.extra_metrics[split]['ddG']['full_log_cosh_loss'] = CustomLogCoshMetric()
            self.extra_metrics[split]['ddG']['log_cosh_loss_only'] = LogCoshOnly()
            self.extra_metrics[split]['ddG']['abs_loss_only'] = AbsLossOnly()

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        
        assert len(batch) == 2
        ddg1, ddg2 = self(batch[0], batch[1])
        true_ddg = batch[0][-1]

        # should equal target ddg
        target_ddg = (ddg1 - ddg2) / 2
        
        # should equal 0
        avg_ddg = (ddg1 + ddg2) / 2
        
        if self.loss == 'mse': # just a lazy MSE type loss
            loss = F.mse_loss(target_ddg, true_ddg) + F.mse_loss(avg_ddg, torch.zeros_like(avg_ddg)) * self.avg_multiplier
        else: # acdc-nn custom loss
            D = target_ddg
            y = true_ddg
            S = avg_ddg
            loss = self.log_cosh(D, y, S)
        
        # # record training/validation metrics
        for metric in self.metrics[f"{prefix}_metrics"]["ddG"].values():
            metric.update(torch.squeeze(target_ddg), torch.squeeze(true_ddg))
            
        # add extra metric (custom loss fxn)
        for extra_metric in self.extra_metrics[f"{prefix}_metrics"]["ddG"].values():
            D = target_ddg
            y = true_ddg
            S = avg_ddg
            extra_metric.update(D, y, S)
            
        on_step = False
        on_epoch = not on_step
        
        output = "ddG"
        for name, metric in self.metrics[f"{prefix}_metrics"][output].items():
            try:
                metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", metric, prog_bar=True, on_step=on_step, on_epoch=on_epoch,
                        batch_size=len(batch))
        
        # extra metric logging
        for name, extra_metric in self.extra_metrics[f"{prefix}_metrics"][output].items():
            try:
                extra_metric.compute()
            except ValueError:
                continue
            self.log(f"{prefix}_{output}_{name}", extra_metric, prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))    
        
        if loss == 0.0:
            return None
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        
        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": self.mpnn_learn_rate}]
        else: # fully frozen MPNN
            param_list = []

        if self.model.lightattn:  # adding light attention parameters
            param_list.append({"params": self.model.light_attention.parameters()})

        mlp_params = [
            {"params": self.model.ddg_out.parameters()}
            ]

        param_list = param_list + mlp_params
        opt = torch.optim.AdamW(param_list, lr=self.learn_rate)

        if self.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, verbose=True, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': 'val_ddG_mse'
            }
        else:
            return opt


def run_siamese_training(cfg):
    """Training function modified for batch-wise siamese network training"""
    print('Running siamese network training instead of standard training loop!')
    
    if 'project' in cfg:
        wandb.init(project=cfg.project, name=cfg.name)
    else:
        cfg.name = 'test'
        
    if 'batch_size' not in cfg.training: # set default batch size to match ProteinMPNN
        cfg.training.batch_size = 10000  # note this is the TOTAL size of the batch (L_max X num_proteins)
    
    
    if len(cfg.datasets) == 1 and cfg.datasets[0].lower() == 'megascale':
        if cfg.training.mutant_structures:
            train_dataset = MegaScaleDatasetSiamesePt(cfg, 'train')
            val_dataset = MegaScaleDatasetSiamesePt(cfg, 'val')            
        else:
            train_dataset = MegaScaleDatasetSiamese(cfg, 'train')
            val_dataset = MegaScaleDatasetSiamese(cfg, 'val')
    else:
        raise ValueError("Batched training not supported for other datasets at this time.")

    if 'num_workers' in cfg.training:
        train_workers, val_workers = int((cfg.training.num_workers - 1) * 0.75), int((cfg.training.num_workers - 1) * 0.25)
    else:
        train_workers, val_workers = 0, 0

    train_loader = DataLoader(train_dataset, collate_fn=None, shuffle=True, num_workers=train_workers, batch_size=None)
    val_loader = DataLoader(val_dataset, collate_fn=None, num_workers=val_workers, batch_size=None)

    model_pl = TransferModelSiamesePL(cfg)

    filename = cfg.name + '_{epoch:02d}_{val_ddG_spearman:.02}'
    monitor = 'val_ddG_spearman'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode='max', dirpath='checkpoints', filename=filename)
    logger = WandbLogger(project=cfg.project, name="test", log_model=False) if 'project' in cfg else None
    max_ep = cfg.training.epochs if 'epochs' in cfg.training else 100

    # TODO fix this - resumed/continued training
    # trainer = pl.Trainer(
    #     callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=200,
    #                      accelerator=cfg.platform.accel, devices=1, ckpt_path='checkpoints/SIAMESE_mse_epoch=99_val_ddG_spearman=0.75.ckpt'
    # )

    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=logger, log_every_n_steps=10, max_epochs=max_ep,
                         accelerator=cfg.platform.accel, devices=1)
    trainer.fit(model_pl, train_loader, val_loader)
    return


def run_prediction_siamese(name, model, dataset_name, dataset, results, keep=False, use_both=True):
    """Standard inference for CSV/PDB based dataset in batched models"""

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    max_batches = None
    metrics = {
        "ddG": get_metrics_new(),
    }
    for m in metrics['ddG'].values():
        m = m.to(device)
    print('Use both?', use_both)
    print('Testing Model %s on dataset %s' % (name, dataset_name))
    preds, ddgs = [], []
    loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=7, batch_size=None)
    for i, batch in enumerate(tqdm(loader)):
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = batch[0]
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
        new_batch_0 = (X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs)

        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs = batch[1]
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
        new_batch_1 = [X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs]

        # pred_fwd, pred_inv = model(new_batch_1, new_batch_0)  # to make lazy reverse mutant dataset
        pred_fwd, pred_inv = model(new_batch_0, new_batch_1)
        if use_both:
            target_ddg = (pred_fwd - pred_inv) / 2  # for true siamese, take avg of both dirs
        else:
            target_ddg = pred_fwd
        sigma = (pred_fwd + pred_inv) / 2
        for metric in metrics["ddG"].values():
            metric.update(torch.squeeze(target_ddg, dim=-1), torch.squeeze(mut_ddGs * -1, dim=-1))

        if max_batches is not None and i >= max_batches:
            break
        
        preds += list(torch.squeeze(target_ddg, dim=-1).detach().cpu())
        ddgs += list(torch.squeeze(mut_ddGs, dim=-1).detach().cpu())

    preds, ddgs = np.squeeze(preds), np.squeeze(ddgs)
    if keep:
        tmp = pd.DataFrame({'ddG_pred': preds, 'ddG_true': ddgs})
        print(tmp.head)
        tmp.to_csv(f'{name}_{dataset_name}_preds_raw_siamese.csv')
    
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


if __name__ == "__main__":
    # for debugging inference step
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cfg = OmegaConf.merge(OmegaConf.load('SIAMESE_lazy.yaml'), OmegaConf.load('local.yaml'))    
   
    models = {
        # "ThermoMPNN-siamese-logcosh": TransferModelSiamesePL.load_from_checkpoint('checkpoints/SIAMESE_paired_epoch=94_val_ddG_spearman=0.67.ckpt', cfg=cfg, map_location=device).model,
        "ThermoMPNN-siamese-MSE": TransferModelSiamesePL.load_from_checkpoint('checkpoints/SIAMESE_mse_epoch=99_val_ddG_spearman=0.75.ckpt', cfg=cfg, map_location=device).model,
        # "ThermoMPNN-siamese-MSE-2": TransferModelSiamesePL.load_from_checkpoint('checkpoints/SIAMESE_mse_2_epoch=71_val_ddG_spearman=0.74.ckpt', cfg=cfg, map_location=device).model,

    }
    
    misc_data_loc = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data'
    datasets = {
        # 'Megascale (test-fakeMUT)': MegaScaleDatasetSiamese(cfg, 'test'),
        # 'Megascale (test-realMUT)': MegaScaleDatasetSiamesePt(cfg, 'test'),

        'Fireprot (HF)': FireProtDatasetSiamese(cfg, 'homologue-free'),

        # 'S669':  ddgBenchDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'S669/pdbs'), csv_fname=os.path.join(misc_data_loc, 'S669/s669_clean_dir.csv'), fwd='xtal', back='xtal-flipped'),
        # 'P53-rev':  ddgBenchDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'), csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv'), 
                                        #    fwd='rosetta', back='rosetta-flipped')
        # 'P53':  ddgBenchDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/P53/pdbs'), csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/P53/p53_clean.csv'), mut_struct=True)
        # 'MYOGLOBIN-rev':  ddgBenchDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'), csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv'), 
                                                #  fwd='rosetta', back='rosetta-flipped')
        # 'MYOGLOBIN':  ddgBenchDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/pdbs'), csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/MYOGLOBIN/myoglobin_clean.csv'), mut_struct=True)

        # 'SSYM-dir': SsymDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'), csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean.csv'), flip=False, mut_struct=False),
        # 'SSYM-inv': SsymDatasetSiamese(cfg, pdb_dir=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/pdbs'), csv_fname=os.path.join(misc_data_loc, 'protddg-bench-master/SSYM/ssym-5fold_clean.csv'), flip=True, mut_struct=False),


    }

    results = []
    for name, model in models.items():
        model = model.eval()
        model = model.to(device)
        for dataset_name, dataset in datasets.items():
            # results = run_prediction_siamese(name, model, dataset_name, dataset, results, keep=False, use_both=True)
            results = run_prediction_siamese(name, model, dataset_name, dataset, results, keep=True, use_both=False)

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("ThermoMPNN_siamese_metrics.csv")
    
    # ------ DEBUGGING -------- #

    # this routine is to prefetch batches for augmentation runs
    # ds = {'Megascale (TR-Siamese-train)': MegaScaleDatasetSiameseAug(cfg, 'train')}
    
    # for dataset_name, dataset in ds.items():
    #     loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=cfg.training.num_workers - 1, batch_size=None)
    #     print('Starting Dataset %s' % dataset_name)
    #     for batch in tqdm(loader):
    #         pass
 
    # ds = {'Megascale (TR-Siamese-val)': MegaScaleDatasetSiameseAug(cfg, 'val')}
    
    # for dataset_name, dataset in ds.items():
    #     loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=cfg.training.num_workers - 1, batch_size=None)
    #     print('Starting Dataset %s' % dataset_name)
    #     for batch in tqdm(loader):
    #         pass

    # ds = {'Megascale (TR-Siamese-test)': MegaScaleDatasetSiameseAug(cfg, 'test')}
    
    # for dataset_name, dataset in ds.items():
    #     loader = DataLoader(dataset, collate_fn=lambda x: x, shuffle=False, num_workers=cfg.training.num_workers - 1, batch_size=None)
    #     print('Starting Dataset %s' % dataset_name)
    #     for batch in tqdm(loader):
    #         pass   
        

