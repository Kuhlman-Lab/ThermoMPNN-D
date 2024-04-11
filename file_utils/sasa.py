import argparse
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import pickle


def main(args):
    parser = PDBParser(QUIET=True)

    struct = parser.get_structure(args.i, args.i)
    model = struct[0]
    dssp = DSSP(model, args.i)
    # save dssp values as pkl
    with open(args.o, 'wb') as fopen:
        pickle.dump(dssp, fopen)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input PDB file')
    parser.add_argument('-o', help='output DSSP .pkl file', default='dssp.pkl')
    main(parser.parse_args())
