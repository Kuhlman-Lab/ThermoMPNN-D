
import bz2
import _pickle as cPickle
from typing import Any

import argparse
import os

import numpy as np
import pandas as pd


def decompress_pickle(file: str) -> Any:
    """
    Load any compressed pickle file.
    Copied from alphafold install
    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def main(args):
    """Parse a directory of AF2 files to extract confidence data"""
    assert os.path.isdir(args.af2_dir)
    files = sorted(os.listdir(args.af2_dir))
    
    af2_pbzs = [f for f in files if ('.pbz' in f) and ('timing' not in f)]    
    names, plddts, ptms, paes = [], [], [], []
    
    for pbz in af2_pbzs:
        
        # get stats from pbz directly
        data = decompress_pickle(os.path.join(args.af2_dir, pbz))
        mean_plddt = np.mean(data['plddt'])
        ptm = data['ptm']
        mean_pae = np.mean(data['pae_output'][0], axis=(0, 1))
        name = pbz.replace('_0_model_1_ptm_0_results.pbz2', '')
        
        names.append(name)
        plddts.append(mean_plddt)
        ptms.append(ptm)
        paes.append(mean_pae)
    
    df = pd.DataFrame({
        'PDB': names, 
        'Mean pLDDT': plddts, 
        'pTM': ptms, 
        'Mean PAE': paes
    })
        
    df.to_csv(args.output)
    
    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--af2_dir', help='directory with af2 outputs to parse', default='./')
    parser.add_argument('--output', help='output csv to save', default='af2_info.csv')
    main(parser.parse_args())