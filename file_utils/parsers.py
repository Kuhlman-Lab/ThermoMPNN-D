from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os


def get_pdb_seq(fname: str, format: str = 'pdb-atom') -> SeqRecord: 
    for record in SeqIO.parse(fname, format):
        seq = str(record.seq)
        if 'X' in seq:
            print('Unknown residue detected!')
            print(fname, '\n', seq)
            if seq.index('X') == 0:  # deal with N-terminal ACE caps by removal
                record.seq = Seq(seq.replace('X', ''))  
            else:
                raise AssertionError('Un-handled residue detected - requires manual correction!')
    return record

def write_seq(fname: str, sequence: SeqRecord, format: str = 'fasta'):
    sequence.description = os.path.basename(fname).strip('.%s' % format)
    SeqIO.write(sequence, fname, format)
    return