import argparse
import pandas as pd
import os
import sys
from urllib import request
import pandas as pd
from tqdm import tqdm


def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    # downloadurl = "https://files.rcsb.org/pub/pdb/data/biounit/PDB/all"
    # downloadurl = "https://files.wwpdb.org/pub/pdb/data/biounit/PDB/all"
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None


def main():
    """Compile all PDBs in protddg-bench datasets and match with csv data."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input protddg csv')
    parser.add_argument('-o', help='output location for saved PDBs')

    args = parser.parse_args()
    df = pd.read_csv(args.i)
    pdbs = df['PDB'].unique()

    if not os.path.isdir(args.o):
        os.makedirs(args.o)
    
    for code in tqdm(pdbs):
        print(code[:4])
        download_pdb(code[:4], args.o)


if __name__ == "__main__":
    main()
