from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import os
import argparse
import pandas as pd
from tqdm import tqdm


def main(args):
    """Calculates DSSP secondary structure assignments for a directory of PDBs and saves them as CSV data."""
    p = PDBParser(QUIET=True)
    
    files = [f for f in os.listdir(args.dir) if f.endswith('.pdb')]
    all_ss = []
    all_fnames = []
    for f in tqdm(files):
        all_fnames.append(f.rstrip('.pdb'))
        full_f =  os.path.join(args.dir, f)
        structure = p.get_structure('f', full_f)[0]
        dssp = DSSP(structure, full_f, file_type='PDB')
        
        sequence = ''
        sec_structure = ''
        for z in range(len(dssp)):
            a_key = list(dssp.keys())[z]
            sequence += dssp[a_key][1]
            sec_structure += dssp[a_key][2]
        
        sec_structure = sec_structure.replace('-', 'C')
        sec_structure = sec_structure.replace('I', 'C')
        sec_structure = sec_structure.replace('T', 'C')
        sec_structure = sec_structure.replace('S', 'C')
        sec_structure = sec_structure.replace('G', 'H')
        sec_structure = sec_structure.replace('B', 'E')
        
        # print(sec_structure)
        all_ss.append(sec_structure)
    
    # all_ss, all_fnames
    df = pd.DataFrame({'PDB': all_fnames, 'DSSP': all_ss})
    df.to_csv(args.out)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', help='directory of PDB files to analyze', default='./')
    parser.add_argument('-out', help='output CSV for results', default='dssp.csv')
    
    main(parser.parse_args())