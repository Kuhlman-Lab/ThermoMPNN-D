from Bio.PDB import PDBParser
import os
import argparse
import pandas as pd
import numpy as np


def main(args):
    """Compiles all sequences from a PDB directory into FASTA format"""
    
    assert os.path.isdir(args.pdbs)
    pdb_list = [s for s in sorted(os.listdir(args.pdbs)) if '.pdb' in s]
    parser = PDBParser(QUIET=True)
    
    print('Parsing %s pdbs...' % (len(pdb_list)))
    methods, resolutions, avg_bfacs = [], [], []
    for pdb in pdb_list:
        structure = parser.get_structure(pdb, os.path.join(args.pdbs, pdb))
        method = structure.header['structure_method']
        resolution = structure.header['resolution']
        methods.append(method.strip())
        resolutions.append(resolution)
        
        if 'x-ray' in method or 'diffraction' in method:  # get avg b factor if available
            model = structure[0]
            atoms = model.get_atoms()
            bfacs = []
            for a in atoms:
                bfacs.append(a.get_bfactor())
            avg_bfacs.append(np.mean(bfacs))
        else:
            avg_bfacs.append(0.)
        
        
    clean_pdb = [s.strip('.pdb') for s in pdb_list]
    df = pd.DataFrame({'PDB': clean_pdb, 
                       'resolution': resolutions, 
                       'method': methods, 
                       'avg b-factor': avg_bfacs})
    
    df.to_csv(args.out)
    print(df.head)
    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='directory of pdbs to read', default='./')
    parser.add_argument('--out', help='output csv to save data', default='./exp_data.csv')    
    main(parser.parse_args())