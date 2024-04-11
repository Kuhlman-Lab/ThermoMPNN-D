# NOTE: taken from https://github.com/dohlee/abyssal-pytorch/

# set location to download/find ESM models
import os
os.environ['TORCH_HOME'] = '/proj/kuhl_lab/users/dieckhau/torch_hub/'

import torch

import pandas as pd
import numpy as np
import sys
sys.path.append('/proj/kuhl_lab/esmfold/esm-main/')

import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

from thermompnn.datasets.v2_datasets import MegaScaleDatasetv2, FireProtDatasetv2, ddgBenchDatasetv2

parser = argparse.ArgumentParser()
parser.add_argument('--local', type=str, default='../../local.yaml', help='local config file to use for ds load')
parser.add_argument('--config', type=str, default='../../infer.yaml', help='config file to use for ds load')
parser.add_argument('--output-dir', '-o', type=str, default='data/embeddings')
parser.add_argument('--batch-size', '-b', type=int, default=496)
parser.add_argument('--repr_layer', type=int, default=6)
parser.add_argument('--label', help='filename label for embeddings', type=str, default='emb')
parser.add_argument('--model', help='name of ESM model to load', default='esm2_t6_8M_UR50D', type=str)
args = parser.parse_args()

# prep dataset - extract WT seqs directly from PDBs
config = OmegaConf.merge(OmegaConf.load(args.local), OmegaConf.load(args.config))

# load dataset and collect PDB-derived sequences
if config.data.dataset.lower() == 'megascale':
    ds = MegaScaleDatasetv2(config, config.data.splits[0])
elif config.data.dataset.lower() == 'fireprot':
    ds = FireProtDatasetv2(config, config.data.splits[0])
elif config.data.dataset.lower() == 'ssym':
    pdb_loc = os.path.join(config.data_loc.misc_data, 'protddg-bench-master/SSYM/pdbs')
    if config.data.splits[0] == 'dir':
        csv_loc = os.path.join(config.data_loc.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_dir.csv')
    else:
        csv_loc = os.path.join(config.data_loc.misc_data, 'protddg-bench-master/SSYM/ssym-5fold_clean_inv.csv')
    ds = ddgBenchDatasetv2(config, pdb_dir=pdb_loc, csv_fname=csv_loc)



pdb_list, seq_list = [], []
for pdb, data in ds.pdb_data.items():
    pdb_list.append(data['name'])
    seq_list.append(data['seq'])

print('Running embeddings for %s sequences with batch size %s' % (str(len(pdb_list)), str(args.batch_size)))

# Load ESM-2 model
if args.model == 'esm2_t6_8M_UR50D':
    from esm.pretrained import esm2_t6_8M_UR50D
    model, alphabet = esm2_t6_8M_UR50D()
elif args.model == 'esm2_t33_650M_UR50D':
    from esm.pretrained import esm2_t33_650M_UR50D
    model, alphabet = esm2_t33_650M_UR50D()
else:
    raise ValueError("Invalid ESM model name %s provided!\tPlease use one of these options: (esm2_t6_8M_UR50D, esm2_t33_650M_UR50D)" % args.model)
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.cuda()
print('Loaded ESM model')

for i in tqdm(range(0, len(pdb_list), args.batch_size)):

    batch = seq_list[i: i + args.batch_size]
    # Get ESM2 embeddings. Set unknown residues to X for now
    data = [(i, seq.replace('-', 'X')) for i, seq in enumerate(batch)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.cuda()
    with torch.no_grad():
        result = model(tokens, repr_layers=[args.repr_layer])
    
    # grab embeddings along variable region indices
    h = result['representations'][args.repr_layer][range(len(batch)), :].cpu().numpy() # [B, L, EMBED_DIM]

    # save embeddings in designated output location    
    wt_names = pdb_list[i: i + args.batch_size]
    for idx, wtn in enumerate(wt_names):
        cpath = args.output_dir
        cpath = os.path.join(cpath, wtn)
        emb = h[idx, ...] # [L, EMBED_DIM]
        print(emb.shape)
        # save individual seqs using WT_name
        fname = cpath + args.label + '.pt'
        print(fname)
        torch.save(emb, fname)
