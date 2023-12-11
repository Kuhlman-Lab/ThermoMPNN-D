import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from math import isnan
from tqdm import tqdm

from thermompnn.protein_mpnn_utils import alt_parse_PDB, parse_PDB
from thermompnn.datasets.dataset_utils import Mutation, seq1_index_to_seq2_index


class MegaScaleDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split):

        self.cfg = cfg
        self.split = split

        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)

        # new type-specific data loading - add option for multi-mutations
        df = df.loc[~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :].reset_index(drop=True)
        mut_list = [df.loc[df.mut_type.str.contains("wt"), :].reset_index(drop=True)] # by default, keep wt for reference
        # mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~(df.mut_type.str.contains("wt")), :].reset_index(drop=True))

        if 'single' in self.cfg.data.mut_types:
            mut_list.append(df.loc[~df.mut_type.str.contains(":") & ~(df.mut_type.str.contains("wt")), :].reset_index(drop=True))
        if 'double' in self.cfg.data.mut_types:
            mut_list.append(df.loc[(df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :].reset_index(drop=True))

        self.df = pd.concat(mut_list, axis=0).reset_index(drop=True)  # this includes points missing structure data

        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split

        if 'reduce' not in cfg:
            cfg.reduce = ''

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = splits['train'] + splits['val'] + splits['test']
            self.wt_names = all_names
        elif self.split == 'train' and 'aug' in self.cfg:  # testing extra PDB data addition
            print('adding extra PDBs to training!')
            full_names = splits['train'] + splits['train_aug']
            self.wt_names = full_names

        else:
            if cfg.reduce == 'prot' and self.split == 'train':
                n_prots_reduced = 58
                self.wt_names = np.random.choice(splits["train"], n_prots_reduced)
            else:
                self.wt_names = splits[self.split]

        self.pdb_data = {}

        # if enabled, will only use PDB ID to match filenames
        self.permissive = self.cfg.data.permissive_pdb if self.cfg.data.permissive_pdb is not None else False
        self.alternate = self.cfg.data.alternate_pdb if self.cfg.data.alternate_pdb is not None else False

        # pre-load all PDBs for faster training (<500 MB total)
        for wt_name in tqdm(self.wt_names):
            wt_rows = self.df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = self.df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            
            # for subsampling the dataset (ablation study)
            if isinstance(cfg.reduce, float) and self.split == 'train':
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(cfg.reduce), replace=False)

            self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
            
            wt_fname = wt_name.removesuffix('.pdb').replace("|",":")

            if self.permissive:
                files = os.listdir(self.cfg.data_loc.megascale_pdbs)
                files = [f for f in files if ('.pdb' in f) and (wt_name in f)]
                if len(files) != 1:  # edge case of duplicate PDB names
                    files = [f for f in files if 'v2' not in f]
                    assert len(files) == 1
                    files = files[0]
                else:
                    files = files[0]
                pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, files)

            # use alternate PDB filenames (used in structure studies)
            elif self.alternate:
                pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name.removeprefix('v2_')}.pdb")
                if not os.path.isfile(pdb_file):  # de novo proteins need to be skipped here
                    return None, None
            else:
                pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_fname}.pdb")
            
            pdb = parse_PDB(pdb_file)
            self.pdb_data[wt_name] = pdb
   
    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""
        if self.alternate:
            return self._alt_batch_retrieval(index)
        
        wt_name = self.wt_names[index]
        mut_data = self.mut_rows[wt_name]

        wt_seq = self.wt_seqs[wt_name]
        pdb = self.pdb_data[wt_name]
        
        assert len(pdb[0]["seq"]) == len(wt_seq)
        pdb[0]["seq"] = wt_seq

        mutations = []
        for i, row in mut_data.iterrows():
            # no insertions or deletions
            if "ins" in row.mut_type or "del" in row.mut_type:
                continue
            # loading double mutants
            elif ":" in row.mut_type:
                assert len(row.aa_seq) == len(wt_seq)
                multi_muts = row.mut_type.split(':')
                wt_list, mut_list, idx_list = [], [], []
                for ml in multi_muts:
                    wt = ml[0]
                    mut = ml[-1]
                    idx = int(ml[1:-1]) - 1
                    assert wt_seq[idx] == wt
                    assert row.aa_seq[idx] == mut
                    wt_list.append(wt)
                    mut_list.append(mut)
                    idx_list.append(idx)
            else:
                assert len(row.aa_seq) == len(wt_seq)
                wt = row.mut_type[0]
                mut = row.mut_type[-1]
                idx = int(row.mut_type[1:-1]) - 1
                assert wt_seq[idx] == wt
                assert row.aa_seq[idx] == mut
                wt_list, mut_list, idx_list = [wt], [mut], [idx]

            if row.ddG_ML == '-':
                continue  # filter out any unreliable data
            ddG = -torch.tensor([float(row.ddG_ML)], dtype=torch.float32)
            mutations.append(Mutation(idx_list, wt_list, mut_list, ddG, wt_name))

        return pdb, mutations

    def _alt_batch_retrieval(self, index):
        """Do batch retreival with built-in sequence alignment for experimental structure processing."""
        wt_name = self.wt_names[index]
        mut_data = self.mut_rows[wt_name]

        wt_seq = self.wt_seqs[wt_name]
        pdb = self.pdb_data[wt_name]
        
        mutations = []
        for i, row in mut_data.iterrows():
            wt = row.mut_type[0]
            mut = row.mut_type[-1]
            idx = int(row.mut_type[1:-1]) - 1
            # CSV data QC
            assert len(row.aa_seq) == len(wt_seq)
            assert wt_seq[idx] == wt
            assert row.aa_seq[idx] == mut
            # need to check and re-align sequences now
            try:
                pdb_idx = idx
                assert pdb[0]['seq'][pdb_idx] == wt == wt_seq[idx]
                
            except (AssertionError, IndexError):  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(wt_seq, pdb[0]['seq'].replace("-", "X"))
                pdb_idx = seq1_index_to_seq2_index(align, idx)
                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == wt == wt_seq[idx]

            wt_list, mut_list, idx_list = [wt], [mut], [pdb_idx]

            if row.ddG_ML == '-':
                continue  # filter out any unreliable data
            ddG = -torch.tensor([float(row.ddG_ML)], dtype=torch.float32)
            mutations.append(Mutation(idx_list, wt_list, mut_list, ddG, wt_name))
        return pdb, mutations


class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, split, model_no = None, pdb_current = None):

        self.cfg = cfg
        self.split = split

        filename = self.cfg.data_loc.fireprot_csv

        df = pd.read_csv(filename).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}
        seq_key = "pdb_sequence"

        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)

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

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        # Note: this is for processing NMR ensembles
        # if model_no is not None:
            # self.model_no = int(model_no)
            # self.pdb_current = pdb_current
        self.df = self.df.loc[self.df['pdb_id_corrected'].isin(self.wt_names)]

        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = df.query('pdb_id_corrected == @wt_name').reset_index(drop=True)
            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]


    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):

        wt_name = self.wt_names[index]
        seq = self.wt_seqs[wt_name]
        data = self.seq_to_data[seq]

        # find pdb file (permissive)
        # pdb_current = self.pdb_current
        # model_no = self.model_no
        # print(pdb_current, wt_name)
        # if pdb_current != wt_name:
            # return None, None
        
        # self.cfg.data_loc.fireprot_pdbs = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/structure_studies/fireprot-HF/NMR_ensembles'
        # files = os.listdir(self.cfg.data_loc.fireprot_pdbs)
        # files = [f for f in files if ('.pdb' in f) and (wt_name in f) and (('_' + str(model_no) + '.pdb') in f)]
        # print(wt_name, files)
        # if len(files) != 1:
        #     files = [f for f in files if 'v2' not in f]
        #     print(wt_name, '***', files)
        #     assert len(files) == 1
        #     files = files[0]
        # else:
        #     files = files[0]
        # pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, files)
        # print('Reading:', wt_name, '-' * 10, '\t', '-' * 10, pdb_file)
        # TODO remove above block once done w/structure studies
    
        pdb_file = os.path.join(self.cfg.data_loc.fireprot_pdbs, f"{data.pdb_id_corrected[0]}.pdb")
        pdb = parse_PDB(self.cfg, pdb_file)

        mutations = []
        for i, row in data.iterrows():
            try:
                pdb_idx = row.pdb_position
                assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
                
            except AssertionError:  # contingency for mis-alignments
                align, *rest = pairwise2.align.globalxx(seq, pdb[0]['seq'].replace("-", "X"))
                pdb_idx = seq1_index_to_seq2_index(align, row.pdb_position)
                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]

            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor([row.ddG], dtype=torch.float32)
            mut = Mutation([pdb_idx], [pdb[0]['seq'][pdb_idx]], [row.mutation], ddG, wt_name)
            mutations.append(mut)

        return pdb, mutations


class ddgBenchDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, pdb_dir, csv_fname, flip=False):

        self.cfg = cfg
        self.pdb_dir = pdb_dir
        self.rev = flip

        df = pd.read_csv(csv_fname)
        self.df = df

        self.wt_seqs = {}
        self.mut_rows = {}
        self.wt_names = df.PDB.unique()

        for wt_name in self.wt_names:
            wt_name_query = wt_name
            wt_name = wt_name[:-1]
            self.mut_rows[wt_name] = df.query('PDB == @wt_name_query').reset_index(drop=True)
            if 'S669' not in self.pdb_dir:
                self.wt_seqs[wt_name] = self.mut_rows[wt_name].SEQ[0]

    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):
        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = [wt_name[-1]]

        wt_name = wt_name.split(".pdb")[0][:-1]
        mut_data = self.mut_rows[wt_name]

        # permissive file searching
        # alt_pdb = alt_parse_PDB(os.path.join('/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/protddg-bench-master/SSYM/pdbs', wt_name + '.pdb'), chain)
        # alt_resn_list = alt_pdb[0]["resn_list"]

        # chain = 'A'
        # self.pdb_dir = '/proj/kuhl_lab/users/dieckhau/ThermoMPNN/data/structure_studies/SSYM/af2/'
        # files = os.listdir(self.pdb_dir)
        # files = [f for f in files if ('.pdb' in f) and (wt_name in f)]
        # print(wt_name, files)
        # if len(files) != 1:
        #     files = [f for f in files if 'v2' not in f]
        #     print(wt_name, '***', files)
        #     assert len(files) == 1
        #     files = files[0]
        # else:
        #     files = files[0]
        # pdb_file = os.path.join(self.pdb_dir, files)
        # print('Reading:', wt_name, '-' * 10, '\t', '-' * 10, pdb_file)
        # pdb = alt_parse_PDB(pdb_file, chain)
        # TODO remove above block once done w/structure studies


        # modified PDB parser returns list of residue IDs so we can align them easier
        pdb_file = os.path.join(self.pdb_dir, wt_name + '.pdb')
        pdb = alt_parse_PDB(pdb_file, chain)
        resn_list = pdb[0]["resn_list"]

        mutations = []
        for i, row in mut_data.iterrows():
            mut_info = row.MUT
            wtAA, mutAA = mut_info[0], mut_info[-1]
            # TODO bring this back after structure_studies are done
            try:
                pos = mut_info[1:-1]
                pdb_idx = resn_list.index(pos)
            except ValueError:  # skip positions with insertion codes for now - hard to parse
                continue
            try:
                assert pdb[0]['seq'][pdb_idx] == wtAA
            except AssertionError:  # contingency for mis-alignments
                # if gaps are present, add these to idx (+10 to get any around the mutation site, kinda a hack)
                if 'S669' in self.pdb_dir:
                    gaps = [g for g in pdb[0]['seq'] if g == '-']
                else:
                    gaps = [g for g in pdb[0]['seq'][:pdb_idx + 10] if g == '-']                

                if len(gaps) > 0:
                    pdb_idx += len(gaps)
                else:
                    pdb_idx += 1
                
                if pdb_idx is None:
                    continue
                assert pdb[0]['seq'][pdb_idx] == wtAA

            # TODO remove after structure_studies
            # need to align alt_pdb to pdb seq now
            # if alt_pdb[0]['seq'][pdb_idx] != pdb[0]['seq'][pdb_idx]:
            #     # do second alignment
            #     align, *rest = pairwise2.align.globalxx(alt_pdb[0]['seq'].replace("-", "X"), pdb[0]['seq'].replace("-", "X"))
            #     pdb_idx = seq1_index_to_seq2_index(align, pdb_idx)
            #     assert pdb[0]['seq'][pdb_idx] == wtAA 

            ddG = None if row.DDG is None or isnan(row.DDG) else torch.tensor([row.DDG * -1.], dtype=torch.float32)
            
            if not self.rev:  # fwd mutations
                mut = Mutation([pdb_idx], [pdb[0]['seq'][pdb_idx]], [mutAA], ddG, wt_name)
                mutations.append(mut)
                
            else: # inverse mutations (w/structure)
                fname = wt_name
                pdb_file = os.path.join(self.pdb_dir, f"{fname}{chain[0]}_{wtAA}{pos}{mutAA}_relaxed.pdb")
                mut_pdb = alt_parse_PDB(pdb_file, chain)
                mut = Mutation([pdb_idx], [mutAA], [wtAA], ddG * -1, row.PDB[:-1])
                mutations.append(mut)
        
        if not self.rev:
            return pdb, mutations
        else:
            return mut_pdb, mutations


class ComboDataset(torch.utils.data.Dataset):
    """For co-training on multiple datasets at once."""
    def __init__(self, cfg, split):

        datasets = []
        if "fireprot" in cfg.datasets:
            fireprot = FireProtDataset(cfg, split)
            datasets.append(fireprot)
        if "megascale" in cfg.datasets:
            mega_scale = MegaScaleDataset(cfg, split)
            datasets.append(mega_scale)
        self.mut_dataset = ConcatDataset(datasets)

    def __len__(self):
        return len(self.mut_dataset)

    def __getitem__(self, index):
        return self.mut_dataset[index]


if __name__ == "__main__":
    # testing functionality
    from omegaconf import OmegaConf
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    cfg = OmegaConf.merge(OmegaConf.load('../../DEBUG.yaml'), OmegaConf.load('../../local.yaml'))   
    
    ds = MegaScaleDataset(cfg, 'test')

    print('Starting dataset iteration')
    for batch in tqdm(ds):
        pass
