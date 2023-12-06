import os
import argparse

from parsers import get_pdb_seq, write_seq

def main(args):
    """Compiles all sequences from a PDB directory into individual FASTA files"""
    
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    
    assert os.path.isdir(args.pdbs)
    pdb_list = [s for s in sorted(os.listdir(args.pdbs)) if '.pdb' in s]
    
    print('Parsing %s pdbs using %s format' % (len(pdb_list), args.fmt))
    for pdb in pdb_list:
        record = get_pdb_seq(os.path.join(args.pdbs, pdb), args.fmt)
        seq_fname = pdb.strip('.pdb') + '.fasta'
        write_seq(os.path.join(args.out, seq_fname), record, 'fasta')

    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='directory of pdbs to read', default='./')
    parser.add_argument('--out', help='output directory for FASTA files', default='./')
    parser.add_argument('--fmt', help='which format to retreive seq (pdb-atom or pdb-seqres)', default='pdb-atom')
    
    main(parser.parse_args())