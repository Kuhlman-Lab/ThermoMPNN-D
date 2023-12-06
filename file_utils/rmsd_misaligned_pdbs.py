import argparse
import os
import pandas as pd
from tqdm import tqdm

from Bio.PDB import PDBParser, Superimposer
from Bio import pairwise2
from parsers import get_pdb_seq


def select_atoms(ref_seq, alt_seq, ref_model, alt_model):
    """Retrieve Ca atoms from a pair of sequences, doing your best to align them along the way."""
    ref_atoms = []
    alt_atoms = []
    
    # detect # of residues in each chain
    assert len(ref_seq) == len(alt_seq)
    for (ref_chain, alt_chain) in zip(ref_model, alt_model):     
        ref_resids = sorted([r.id[1] for r in ref_chain.get_residues()])
        alt_resids = sorted([r.id[1] for r in alt_chain.get_residues()])

        global_inc, ref_inc, alt_inc = 0, 0, 0
        while True:
            if global_inc >= len(ref_seq):
                break
            # check if residue is '-' - if so, skip along
            # this is the increment along the aligned sequences - shared b/w both
            ref_aa_aligned = ref_seq[global_inc]
            alt_aa_aligned = alt_seq[global_inc]
            print(ref_aa_aligned, alt_aa_aligned, global_inc, len(ref_seq), len(alt_seq))
            alt_flag, ref_flag = False, False
            
            try:
                if ref_aa_aligned == '-' or ref_aa_aligned == 'X':
                    pass
                    # aa is missing - skip it, but don't increment the PDB seq
                else:
                    # aa is present - grab it and increment the PDB seq      
                    if ref_inc >= len(ref_resids):
                        break        
                    r = ref_chain[ref_resids[ref_inc]]['CA']
                    ref_inc += 1
                    ref_flag = True
                    
                if alt_aa_aligned == '-' or alt_aa_aligned == 'X':
                    pass
                else:
                    a = alt_chain[alt_resids[alt_inc]]['CA']
                    alt_inc += 1
                    alt_flag = True
            
            except KeyError:  # if either one is missing CA atom, drop both and iterate past it
                if ref_flag:
                    print('Alt CA missing')
                    ref_inc -= 1
                    ref_flag = False
                else:
                    print('Ref CA missing')
                    ref_inc += 1
                    # alt_inc += 1
                global_inc += 1
                continue
            # check if both residues were read
            if alt_flag and ref_flag and (ref_aa_aligned == alt_aa_aligned):
                ref_atoms.append(r)
                alt_atoms.append(a)
            global_inc += 1
            if global_inc >= len(ref_seq):
                break

    assert len(ref_atoms) == len(alt_atoms)
    return ref_atoms, alt_atoms

def calculate_RMSD_from_atoms(ref, alt, superimposer):
    """Calculate pairwise Ca-Ca RMSD for two proteins"""
    superimposer.set_atoms(ref, alt)
    superimposer.apply(ref)
    return superimposer.rms

def main(args):
    """Calculates pairwise RMSD for mismatched pdb structure directories using filename matching"""
    exps = [s for s in os.listdir(args.exp) if '.pdb' in s]
    af2s = [s for s in os.listdir(args.af2) if '.pdb' in s]
    # there are fewer exps than af2s - use exps as guide
    
    parser = PDBParser(QUIET=True)
    super_imposer = Superimposer()
    
    
    af2s_full = af2s
    rmsds, ca_matched, prefixes = [], [], []
    for e in tqdm(exps):
        prefix = e.removesuffix('.pdb')
        prefixes.append(prefix)
        af2s = [s for s in af2s_full if s.startswith(prefix)]
        print(prefix, af2s)
        if len(af2s) == 0:
            af2s = [s for s in af2s_full if s.startswith('v2_' + prefix)]
        assert len(af2s) == 1
        af2 = af2s[0]
        
        exp_record = get_pdb_seq(os.path.join(args.exp, e), 'pdb-atom')
        af2_record = get_pdb_seq(os.path.join(args.af2, af2), 'pdb-atom')
        
        align, *rest = pairwise2.align.globalxx(str(exp_record.seq), str(af2_record.seq))

        exp_pdb = parser.get_structure('exp', os.path.join(args.exp, e))[0]
        af2_pdb = parser.get_structure('exp', os.path.join(args.af2, af2))[0]

        exp_aligned, af2_aligned = align.seqA, align.seqB
        exp_atoms, af2_atoms = select_atoms(exp_aligned, af2_aligned, exp_pdb, af2_pdb)
        rmsd = calculate_RMSD_from_atoms(exp_atoms, af2_atoms, super_imposer)

        rmsds.append(rmsd)
        ca_matched.append(len(exp_atoms))
    
    df = pd.DataFrame({'PDB': prefixes, 'Exp-vs-AF2 RMSD': rmsds, 'Ca Atoms Matched': ca_matched})
    df.to_csv(args.output)
    print(df.head)
    return




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='directory of experimental PDBs to calculate', default='./')
    parser.add_argument('--af2', help='directory for af2 structures to use', default='./')
    parser.add_argument('--output', help='output csv to save', default='./exp_vs_af2_rmsds.csv')
    
    main(parser.parse_args())