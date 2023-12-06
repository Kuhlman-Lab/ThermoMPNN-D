import argparse
import os
import sys
from tqdm import tqdm

def main(args):
    """Reads a directory of PDBs and auto-downloads PDBs based on the filenames"""
    
    files = [f for f in sorted(os.listdir(args.input)) if '.pdb' in f]
    files = [f.rstrip('.pdb') for f in files]
    # filter out de novo designs
    files = [f for f in files if ('rd' not in f) and ('Hall' not in f)]
    # handle "v2" labeled PDBs
    files = [f.removeprefix('v2').removeprefix('_') for f in files]
    files = [f for f in files if '_' not in f]
    print('Parsing %s files' % len(files))
    print(files)
    
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    from download_protddg_pdbs import download_pdb
    for f in tqdm(files):
        download_pdb(f, args.output)
    
    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='directory of files to check for PDB IDs', default='./')
    parser.add_argument('--output', help='directory to save retrieved PDBs to', default='./')
    
    main(parser.parse_args())