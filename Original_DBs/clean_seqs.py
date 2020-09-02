#!/bin/env pyhton

import numpy as np
from numpy import save
import sys
from Bio import SeqIO

"""
This function takes as input a sequence of nucleotides and cleans it
of characters other than A, C, G, T or N
"""
def filterSeq(file):
	cleanFile = open(file+".clean", "w")
	langu = ['A', 'C', 'G', 'T', 'N']
	for te in SeqIO.parse(file, "fasta"):
		seq = str(te.seq)
		newSeq = [x for x in seq if x.upper() in langu]
		cleanFile.write(">"+str(te.id)+"\n"+"".join(newSeq)+"\n")

if __name__ == '__main__':

	seqfile = sys.argv[1]
	filterSeq(seqfile)

