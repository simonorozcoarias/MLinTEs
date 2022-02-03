#!/bin/env pyhton

import numpy as np
from numpy import save
import sys
from Bio import SeqIO


def numSubSeqs(file, minlength):
	count = 0
	for te in SeqIO.parse(file, "fasta"):
		for n in range(0, len(te.seq), minlength):
			count += 1
	print("total subseqs: "+str(count))
	return count

"""
This function takes as input classified TEs at lineage level and transforms nucleotide sequences in
a 2D representation using one hot coding (methods 1 to 5)
"""
def conversion2d(file, minlength):

	numSeqs = numSubSeqs(file, minlength)
	langu = ['A', 'C', 'G', 'T', 'N']
	rep2d = np.zeros((numSeqs, 5, minlength), dtype=np.int16)
	posSeq = 0
	labels = np.zeros((numSeqs, 1), dtype=np.int16)

	print("Doing conversion into 2D of "+file+" using shortest method")

	for te in SeqIO.parse(file, "fasta"):
		seq = str(te.seq)
		posNucl = 0
		order = -1
		if str(te.id).upper().find("_NEGATIVA") != -1:
			order = 0
		elif str(te.id).upper().find("-ALE") != -1 or str(te.id).upper().find("RETROFIT-") != -1:
			order = 1
		# elif str(te.id).upper().find("ALESIA-") != -1:
		# 	order = 2
		elif str(te.id).upper().find("-ANGELA") != -1:
			order = 3
		elif str(te.id).upper().find("-BIANCA") != -1:
			order = 4
		# elif str(te.id).upper().find("BRYCO-") != -1:
		# 	order = 5
		# elif str(te.id).upper().find("LYCO-") != -1:
		# 	order = 6
		# elif str(te.id).upper().find("GYMCO-") != -1:
		# 	order = 7
		elif str(te.id).upper().find("-IKEROS") != -1:
			order = 8
		elif str(te.id).upper().find("-IVANA") != -1 or str(te.id).upper().find("-ORYCO") != -1:
			order = 9
		# elif str(te.id).upper().find("OSSER-") != -1:
		# 	order = 10
		# elif str(te.id).upper().find("TAR-") != -1:
		# 	order = 11
		elif str(te.id).upper().find("-TORK") != -1:
			order = 12
		elif str(te.id).upper().find("-SIRE") != -1:
			order = 13
		elif str(te.id).upper().find("-CRM") != -1:
			order = 14
		# elif str(te.id).upper().find("CHLAMYVIR-") != -1:
		# 	order = 15
		elif str(te.id).upper().find("-GALADRIEL") != -1:
			order = 16
		elif str(te.id).upper().find("-REINA") != -1:
			order = 17
		elif str(te.id).upper().find("-TEKAY") != -1 or str(te.id).upper().find("-DEL") != -1:
			order = 18
		elif str(te.id).upper().find("-ATHILA") != -1:
			order = 19
		elif str(te.id).upper().find("-TAT") != -1:
			order = 20
		elif str(te.id).upper().find("-OGRE") != -1:
			order = 21
		elif str(te.id).upper().find("-RETAND") != -1:
			order = 22
		# elif str(te.id).upper().find("PHYGY-") != -1:
		# 	order = 23
		# elif str(te.id).upper().find("SELGY-") != -1:
		# 	order = 24
		if order != -1:	
			if len(seq) <= minlength:
				for nucl in seq:
					posLang = langu.index(nucl.upper())
					rep2d[posSeq][posLang][posNucl] = 1
					posNucl += 1
				labels[posSeq][0] = order
				posSeq += 1
			else:
				for n in range(0, len(seq), minlength):
					posNucl = 0
					if n+minlength > len(seq):
						actualEnd = len(seq)
						actualStart = len(seq) - minlength
					else:
						actualEnd = n + minlength
						actualStart = n
					newSeq = seq[actualStart: actualEnd]
					for nucl in newSeq:
						posLang = langu.index(nucl.upper())
						rep2d[posSeq][posLang][posNucl] = 1
						posNucl += 1
					labels[posSeq][0] = order
					posSeq += 1
		else:
			print("---------- error: --------------")
			print(te.id)
			print("--------------------------------")

		if posSeq % 500 == 0:
			print("Doing "+str(posSeq) +" of "+str(numSeqs))

	print("saving features file...")
	save(file+'_shortest.npy', rep2d.astype(np.int8))
	print("done!!")
	print("saving labels file...")
	save(file+'_shortest_labels.npy', labels.astype(np.int8))
	print("done!!")

"""
This function calculates the minimum length found in the dataset
"""
def minLength(file):
	minLen = 50000
	numSeqs = 0
	for te in SeqIO.parse(file, "fasta"):
		if len(te.seq) < minLen:
			minLen = len(te.seq)
		numSeqs += 1
	return (minLen, numSeqs)

"""
This function deletes all characters that are no DNA (A, C, G, T, N)
"""
def filter(file):
	newFile = open(file+".filtered", "w")
	for te in SeqIO.parse(file, "fasta"):
		seq = str(te.seq)
		filterDna = [x for x in seq if x.upper() in ['A', 'C', 'G', 'T', 'N']]
		newSeq = "".join(filterDna)
		newFile.write(">"+str(te.id)+"\n"+newSeq+"\n")


if __name__ == '__main__':
	seqfile = sys.argv[1]
	filter(seqfile)
	#minLen, numSeqs = minLength(seqfile+".filtered")
	#print(minLen)
	conversion2d(seqfile+".filtered", 1000)

