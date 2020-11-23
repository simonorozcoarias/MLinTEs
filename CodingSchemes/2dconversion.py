#!/bin/env pyhton

import numpy as np
from numpy import save
import sys
from Bio import SeqIO

"""
This function takes as input classified TEs at lineage level and transforms nucleotide sequences in
a 2D representation using one hot coding (methods 1 to 5)
"""
def conversion2d(file, maxlength, numSeqs, method):

	langu = ['A', 'C', 'G', 'T', 'N']
	rep2d = np.zeros((numSeqs, 5, maxlength))
	posSeq = 0
	labels = np.zeros((numSeqs, 1))
	methodName = ""

	if method == 1:
		# to complete TEs with self-replication method
		methodName = "self"
	elif method == 2:
		# to complete TEs with zeros, it's not necessary doing anything
		methodName = "zeros"	
	elif method == 3:
		# to complete TEs with NNs
		methodName = "NNs"
	elif method == 4:
		# to complete TEs with NNs centering the sequence
		methodName = "center"
	elif method == 5:
		# to complete TEs with ones
		methodName = "ones"

	print("Doing conversion into 2D of "+file+" using "+methodName+" method")

	for te in SeqIO.parse(file, "fasta"):
		seq = str(te.seq)
		posNucl = 0

		order = -1
		if str(te.id).upper().find("_NEGATIVA") != -1:
			order = 0
		elif str(te.id).upper().find("ALE-") != -1 or str(te.id).upper().find("RETROFIT-") != -1:
			order = 1
		# elif str(te.id).upper().find("ALESIA-") != -1:
		# 	order = 2
		elif str(te.id).upper().find("ANGELA-") != -1:
			order = 3
		elif str(te.id).upper().find("BIANCA-") != -1:
			order = 4
		# elif str(te.id).upper().find("BRYCO-") != -1:
		# 	order = 5
		# elif str(te.id).upper().find("LYCO-") != -1:
		# 	order = 6
		# elif str(te.id).upper().find("GYMCO-") != -1:
		# 	order = 7
		elif str(te.id).upper().find("IKEROS-") != -1:
			order = 8
		elif str(te.id).upper().find("IVANA-") != -1 or str(te.id).upper().find("ORYCO-") != -1:
			order = 9
		# elif str(te.id).upper().find("OSSER-") != -1:
		# 	order = 10
		# elif str(te.id).upper().find("TAR-") != -1:
		# 	order = 11
		elif str(te.id).upper().find("TORK-") != -1:
			order = 12
		elif str(te.id).upper().find("SIRE-") != -1:
			order = 13
		elif str(te.id).upper().find("CRM-") != -1:
			order = 14
		# elif str(te.id).upper().find("CHLAMYVIR-") != -1:
		# 	order = 15
		elif str(te.id).upper().find("GALADRIEL-") != -1:
			order = 16
		elif str(te.id).upper().find("REINA-") != -1:
			order = 17
		elif str(te.id).upper().find("TEKAY-") != -1 or str(te.id).upper().find("DEL-") != -1:
			order = 18
		elif str(te.id).upper().find("ATHILA-") != -1:
			order = 19
		elif str(te.id).upper().find("TAT-") != -1:
			order = 20
		elif str(te.id).upper().find("OGRE-") != -1:
			order = 21
		elif str(te.id).upper().find("RETAND-") != -1:
			order = 22
		# elif str(te.id).upper().find("PHYGY-") != -1:
		# 	order = 23
		# elif str(te.id).upper().find("SELGY-") != -1:
		# 	order = 24
		if order != -1:	
			if len(seq) < maxlength:
					if method == 1:
						# to complete TEs with self-replication method
						times = int((maxlength-len(seq))/len(seq))+1
						seq = str(seq+(str(seq)*(times+1)))[0:maxlength]	
					#elif method == 2:
						# to complete TEs with zeros, it's not necessary doing anything
					elif method == 3:
						# to complete TEs with NNs
						times = maxlength-len(seq)
						NNs = 'N'*(times+1)
						seq = str(str(seq)+NNs)[0:maxlength]
					elif method == 4:
						# to complete TEs with NNs centering the sequence
						times = int((maxlength-len(seq))/2)
						seq = str('N'*times+str(seq)+'N'*(times+1))[0:maxlength]
			for nucl in seq:
				posLang = langu.index(nucl.upper())
				rep2d[posSeq][posLang][posNucl] = 1
				posNucl += 1

			# to complete TEs with ones
			if method == 5:
				for i in range(len(seq), maxlength):
					rep2d[posSeq][0][i] = 1
					rep2d[posSeq][1][i] = 1
					rep2d[posSeq][2][i] = 1
					rep2d[posSeq][3][i] = 1
					rep2d[posSeq][4][i] = 1

			labels[posSeq][0] = order
		else:
			print("---------- error: --------------")
			print(te.id)
			print("--------------------------------")

		if posSeq % 500 == 0:
			print("Doing "+str(posSeq) +" of "+str(numSeqs))
		posSeq += 1

	print("saving features file...")
	save(file+'_'+methodName+'.npy', rep2d.astype(np.int8))
	print("done!!")
	print("saving labels file...")
	save(file+'_'+methodName+'_labels.npy', labels.astype(np.int8))
	print("done!!")


"""
This function takes as input classified TEs at lineage level and transforms nucleotide sequences in
a 2D representation using one hot coding (as implemented in TERL)
"""
def conversion2dTERL(file, maxlength, numSeqs, method):

	langu = ['A', 'C', 'G', 'T', 'N']
	rep2d = np.zeros((numSeqs, 6, maxlength))
	posSeq = 0
	labels = np.zeros((numSeqs, 1))

	print("Doing conversion into 2D of "+file+" using TERL method")

	for te in SeqIO.parse(file, "fasta"):
		seq = str(te.seq)
		posNucl = 0

		order = -1
		if str(te.id).upper().find("_NEGATIVA") != -1:
			order = 0
		elif str(te.id).upper().find("ALE-") != -1 or str(te.id).upper().find("RETROFIT-") != -1:
			order = 1
		# elif str(te.id).upper().find("ALESIA-") != -1:
		# 	order = 2
		elif str(te.id).upper().find("ANGELA-") != -1:
			order = 3
		elif str(te.id).upper().find("BIANCA-") != -1:
			order = 4
		# elif str(te.id).upper().find("BRYCO-") != -1:
		# 	order = 5
		# elif str(te.id).upper().find("LYCO-") != -1:
		# 	order = 6
		# elif str(te.id).upper().find("GYMCO-") != -1:
		# 	order = 7
		elif str(te.id).upper().find("IKEROS-") != -1:
			order = 8
		elif str(te.id).upper().find("IVANA-") != -1 or str(te.id).upper().find("ORYCO-") != -1:
			order = 9
		# elif str(te.id).upper().find("OSSER-") != -1:
		# 	order = 10
		# elif str(te.id).upper().find("TAR-") != -1:
		# 	order = 11
		elif str(te.id).upper().find("TORK-") != -1:
			order = 12
		elif str(te.id).upper().find("SIRE-") != -1:
			order = 13
		elif str(te.id).upper().find("CRM-") != -1:
			order = 14
		# elif str(te.id).upper().find("CHLAMYVIR-") != -1:
		# 	order = 15
		elif str(te.id).upper().find("GALADRIEL-") != -1:
			order = 16
		elif str(te.id).upper().find("REINA-") != -1:
			order = 17
		elif str(te.id).upper().find("TEKAY-") != -1 or str(te.id).upper().find("DEL-") != -1:
			order = 18
		elif str(te.id).upper().find("ATHILA-") != -1:
			order = 19
		elif str(te.id).upper().find("TAT-") != -1:
			order = 20
		elif str(te.id).upper().find("OGRE-") != -1:
			order = 21
		elif str(te.id).upper().find("RETAND-") != -1:
			order = 22
		# elif str(te.id).upper().find("PHYGY-") != -1:
		# 	order = 23
		# elif str(te.id).upper().find("SELGY-") != -1:
		# 	order = 24
		if order != -1:	
			for nucl in seq:
				posLang = langu.index(nucl.upper())
				rep2d[posSeq][posLang][posNucl] = 1
				posNucl += 1

			# to fill the border
			for i in range(len(seq), maxlength):
				rep2d[posSeq][5][i] = 1

			labels[posSeq][0] = order
		else:
			print("---------- error: --------------")
			print(te.id)
			print("--------------------------------")

		if posSeq % 500 == 0:
			print("Doing "+str(posSeq) +" of "+str(numSeqs))
		posSeq += 1

	print("saving features file...")
	save(file+'_TERL.npy', rep2d)
	print("done!!")
	print("saving labels file...")
	save(file+'_TERL_labels.npy', labels)
	print("done!!")

"""
This function calculates the maximum length found in the dataset
"""
def maxLength(file):
	maxLen = 0
	numSeqs = 0
	for te in SeqIO.parse(file, "fasta"):
		if len(te.seq) > maxLen:
			maxLen = len(te.seq)
		numSeqs += 1
	return (maxLen, numSeqs)

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
	method = int(sys.argv[2])
	filter(seqfile)
	maxLen, numSeqs = maxLength(seqfile+".filtered")
	print(maxLen)
	if method <= 5:
		conversion2d(seqfile+".filtered", maxLen, numSeqs, method)
	elif method == 6: # TERL method:
		conversion2dTERL(seqfile+".filtered", maxLen, numSeqs, method)

