from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'


@dataclass
class Mutation:
    """Custom dataclass for holding key mutation information"""
    position: list[int]
    wildtype: list[str]
    mutation: list[str]
    ddG: Optional[float] = None
    pdb: Optional[str] = ''


def seq1_index_to_seq2_index(align, index):
    """Given an alignment object and an index in seqA, return new index in seqB."""
    cur_seq1_index = 0

    # first find the aligned index
    for aln_idx, char in enumerate(align.seqA):
        if char != '-':
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break
    
    # now the index in seq 2 cooresponding to aligned index
    if align.seqB[aln_idx] == '-':
        return None

    seq2_to_idx = align.seqB[:aln_idx+1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == '-':
            seq2_idx -= 1
    
    if seq2_idx < 0:
        return None

    return seq2_idx

