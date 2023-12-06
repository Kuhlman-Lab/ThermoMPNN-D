import pandas as pd
import os
import torch
from omegaconf import OmegaConf
from Bio.PDB import PDBParser

import sys
sys.path.append('../')
sys.path.append('./')
from protein_mpnn_utils import parse_PDB
from thermompnn_benchmarking import get_trained_model

import time
from dataclasses import dataclass
from typing import Optional


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


@dataclass
class Mutation:
    position: list[int]
    wildtype: list[str]
    mutation: list[str]
    chain: list[str]
    chain_position: list[int]
    ddG: Optional[float] = None
    pdb: Optional[str] = ''


def get_chains(pdb):
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure('', pdb)
  chains = [c.id for c in structure.get_chains()]
  return chains


def get_ssm_mutations(pdb):
    # make mutation list for SSM run - use ALL chains
    mutation_list = []
    chain_seqs = [k for k in pdb.keys() if k.startswith('seq_chain_')]
    total_seq = pdb['seq']
    total_offset = 0
    for cs in chain_seqs:
        seq = pdb[cs]
        for seq_pos in range(len(seq)):
            assert total_seq[total_offset + seq_pos] == seq[seq_pos]
            wtAA = seq[seq_pos]
            # check for missing residues
            if wtAA != '-':
                # add each mutation option
                for mutAA in ALPHABET[:-1]:
                    mutation_list.append(Mutation(position=[int(total_offset + seq_pos)], 
                                                  wildtype=[wtAA], 
                                                  mutation=[mutAA], 
                                                  chain=[cs[-1]], 
                                                  chain_position=[int(seq_pos)]))
                    # mutation_list.append(wtAA + str(total_offset + seq_pos) + mutAA)
            else:
                mutation_list.append(None)
        total_offset += len(seq)
    return mutation_list



def main(cfg, args):

    # define config for model loading
    config = {
        'training': {
            'num_workers': 8,
            'learn_rate': 0.001,
            'epochs': 100,
            'lr_schedule': True,
        },
        'data': {
            'TR': False,
            'mut_types': ['single'],
        },
        'model': {
            'hidden_dims': [64, 32],
            'subtract_mut': True,
            'num_final_layers': 2,
            'freeze_weights': True,
            'load_pretrained': True,
            'lightattn': True,
            'lr_schedule': True,
        }
    }

    cfg = OmegaConf.merge(config, cfg)

    # load the chosen model and dataset
    models = {
        "ThermoMPNN": get_trained_model(model_name=args.model_path,
                                        config=cfg, override_custom=True)
    }

    input_pdb = args.pdb
    pdb_id = os.path.basename(input_pdb).rstrip('.pdb')

    datasets = {
        pdb_id: args.pdb
    }

    raw_pred_df = pd.DataFrame(columns=['Model', 'Dataset', 'ddG_pred', 'position', 'wildtype', 'mutation',])
    row = 0
    for name, model in models.items():
        model = model.eval()
        model = model.cuda()
        for dataset_name, dataset in datasets.items():
            prep_start = time.time()
            if len(args.chain) < 1:  # if unspecified, take ALL chains
                chain = get_chains(input_pdb)
            else:
                chain = args.chain
            print(input_pdb, chain)
            mut_pdb = parse_PDB(input_pdb, chain)
            mutation_list = get_ssm_mutations(mut_pdb[0])

            prep_end = time.time()
            inf_start = time.time()
            print(mutation_list)
            print('Calculating %s predictions for PDB %s' % (str(len(mutation_list)), mutation_list[0].pdb.strip('.pdb')))
            pred, _ = model(mut_pdb, mutation_list)
            inf_end = time.time()

            for mut, out in zip(mutation_list, pred):
                if mut is not None:
                    col_list = ['ddG_pred', 'position', 'wildtype', 'mutation', 'pdb', 'chain', 'chain_position']
                    val_list = [out["ddG"].cpu().item(), mut.position[0] + 1, mut.wildtype[0],
                                mut.mutation[0], mut.pdb.strip('.pdb'), mut.chain[0], mut.chain_position[0] + 1]
                    for col, val in zip(col_list, val_list):
                        raw_pred_df.loc[row, col] = val

                    raw_pred_df.loc[row, 'Model'] = name
                    raw_pred_df.loc[row, 'Dataset'] = dataset_name
                    row += 1
            
            prep_time = round(prep_end - prep_start, 3)
            print(f'Preprocessing time: {prep_time} seconds')
            inf_time = round(inf_end - inf_start, 3)
            print(f'Inference time: {inf_time} seconds')

    print(raw_pred_df)
    raw_pred_df.to_csv("ThermoMPNN_inference_%s.csv" % pdb_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default='', help='Input PDB to use for custom inference')
    parser.add_argument('--chain', type=list, nargs='+', default=[], help='Chain in input PDB to use.')
    parser.add_argument('--model_path', type=str, default='../models/thermoMPNN_default.py', help='filepath to model to use for inference')

    args = parser.parse_args()
    cfg = OmegaConf.load("../local.yaml")
    with torch.no_grad():
        main(cfg, args)
