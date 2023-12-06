import os
import argparse
import pickle
import shutil


def main(args):
    """Sort PDBs according to a specified splits file full of filenames."""
    with open(args.splits, 'rb') as f:
        splits = pickle.load(f)
    
    pdb_names = splits[args.fold]
    pdb_names = [p.rstrip('.pdb').replace("|", ":") for p in pdb_names]

    pdbs_available = [p.rstrip('.pdb').replace("|", ":") for p in os.listdir(args.pdbs)]

    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)
        
    print('PDBs to fetch:', len(pdb_names))
    n = 0
    for name in pdb_names:
        if name in pdbs_available:
            n += 1
            shutil.copy2(
                os.path.join(args.pdbs, name + '.pdb'), 
                os.path.join(args.dest, name + '.pdb')
            )

    print('Matches found: ', n, '\t of \t', len(pdb_names))
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='pdb folder to check')
    parser.add_argument('--dest', help='destination folder to use')
    parser.add_argument('--splits', help='splits file to use')
    parser.add_argument('--fold', help='which split/fold to use in the splits file')

    args = parser.parse_args()
    main(args)

