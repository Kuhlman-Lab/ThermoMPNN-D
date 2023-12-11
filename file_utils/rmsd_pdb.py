import argparse
from Bio.PDB import PDBParser, Superimposer


def select_atoms(ref_model, alt_model, ca=True):
    """Retrieve atoms from a pair of aligned sequences"""
    ref_atoms = []
    alt_atoms = []
    
    # detect # of residues in each chain
    for (ref_chain, alt_chain) in zip(ref_model, alt_model):     
        for (ref_res, alt_res) in zip(ref_chain, alt_chain):
            if ca:
                try:
                    r = ref_res['CA']
                    a = alt_res['CA']
                    ref_atoms.append(r)
                    alt_atoms.append(a)
                except KeyError:
                    print('Missing CA, skipping residue')
            else:
                for (ref_atom, alt_atom) in zip(ref_res, alt_res):
                    ref_atoms.append(ref_atom)
                    alt_atoms.append(alt_atom)

    if len(ref_atoms) != len(alt_atoms):
        raise AssertionError("Structures have different numbers of atoms! Alignment Failed.")
    return ref_atoms, alt_atoms

def calculate_RMSD_from_atoms(ref, alt, superimposer):
    """Calculate pairwise RMSD for two proteins"""
    superimposer.set_atoms(ref, alt)
    superimposer.apply(ref)
    return superimposer.rms

def main(args):
    """Calculates pairwise RMSD for two aligned PDBs"""
    
    pdb1, pdb2 = args.pdb1, args.pdb2
    
    if not (pdb1.endswith('.pdb') and pdb2.endswith('.pdb')):
        raise AssertionError("Files do not end with .pdb !")
    
    parser = PDBParser(QUIET=True)
    super_imposer = Superimposer()
    
    struct1 = parser.get_structure(pdb1, pdb1)[0]
    struct2 = parser.get_structure(pdb2, pdb2)[0]

    ref_atoms, alt_atoms = select_atoms(struct1, struct2, ca=(not args.full))

    rmsd = calculate_RMSD_from_atoms(ref_atoms, alt_atoms, super_imposer)

    print(f'RMSD of {rmsd} A calculated from {len(ref_atoms)} paired atoms')

    return rmsd


if __name__ == "__main__":
    """
    Pairwise PDB RMSD script.
    NOTE: This only works for structures with an equal number of atoms. For partial alignments, an error will be thrown.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb1', help='pdb1 filename')
    parser.add_argument('--pdb2', help='pdb2 filename')
    parser.add_argument('--full', action='store_true', help='do full-atom alignment (default is Calpha-only)')
    main(parser.parse_args())