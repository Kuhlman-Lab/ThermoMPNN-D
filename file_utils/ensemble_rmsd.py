import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from Bio.PDB import PDBParser, Superimposer


def get_Ca_atoms(ref_model, alt_model):
    """Retrieve Ca atoms"""
    ref_atoms = []
    alt_atoms = []
    for (ref_chain, alt_chain) in zip(ref_model, alt_model):
        for ref_res, alt_res in zip(ref_chain, alt_chain):
            try:
                r = ref_res['CA']
                a = alt_res['CA']
            except KeyError:
                continue
            ref_atoms.append(r)    
            alt_atoms.append(a)
    return ref_atoms, alt_atoms

def calculate_rmsd(prot1, prot2, pdbparser, superimposer, pdb_dir):
    """Calculate pairwise Ca-Ca RMSD for two proteins"""
    structure1 = pdbparser.get_structure('ref', os.path.join(pdb_dir, prot1))[0]
    structure2 = pdbparser.get_structure('alt', os.path.join(pdb_dir, prot2))[0]
    ref, alt = get_Ca_atoms(structure1, structure2)
    superimposer.set_atoms(ref, alt)
    # print(prot1, prot2, superimposer.rms)
    return superimposer.rms

def main(args):
    """Calculates all-vs-all RMSD for NMR ensemble models"""
    pdbs = [s for s in os.listdir(args.pdbs) if '.pdb' in s]
    prefixes = np.unique([p.split('_')[0] for p in pdbs])

    parser = PDBParser(QUIET=True)
    super_imposer = Superimposer()

    single_rms, members = [], []

    for pre in tqdm(prefixes):
        ensemble_members = [p for p in pdbs if pre in p]
        rms_all = []
        print('%s ensemble members for protein %s' % (len(ensemble_members), pre))
        for ens1 in ensemble_members:
            for ens2 in ensemble_members:
                if ens1 != ens2:
                    rms_all.append(calculate_rmsd(ens1, ens2, parser, super_imposer, args.pdbs))
        single_rms.append(np.mean(rms_all))
        print(round(np.mean(rms_all), 4))
        members.append(len(ensemble_members))
    
    df = pd.DataFrame({'PDB': prefixes, 'Ensemble RMSD': single_rms, 'Ensemble Members': members})
    df.to_csv(args.output)
    print(df.head)

    return


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdbs', help='directory of PDBs to calculate', default='./')
    parser.add_argument('--output', help='output csv to save', default='./ensemble_rmsd.csv')
    
    main(parser.parse_args())