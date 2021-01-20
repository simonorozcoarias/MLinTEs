#!/bin/env pyhton

import numpy as np
from numpy import save
import sys
import os
from Bio import SeqIO
import multiprocessing
import itertools
import time


"""
This function takes as input classified TEs at lineage level and transforms nucleotide sequences in
a 2D representation using one hot coding (as implemented in TERL)
"""
def positional_kmers(file, maxlength, seqs_per_procs, id, remain, TEids, TEseqs, numSeqs):

	langu = ['A','T','AGGGGG','CCCCCT','CATC','ACATA','TATGT','AGCT','CCTT','ATAT','AAGG','C','G','AAAAAA','GATGA','TTTTTT','CTT','ACAT','AGA','ATATA','ATGT','AAG','TGG','TTAA','TTC','GATC','TATAT','CTAG','GAA','CCA','CTTT','GAAG','CCAAT','TCTC','GTAC','CTTG','GAGA','TATG','CCTTT','GGGGGA','TAAC','TGCC','CCAT','AAAGG','TCCCCC','TCATC','GATG','GGCC','TGTA','AAGC','TACC','GGGGAG','CTTC','TTAAT','GATTGG','AACTC','AATC','GAGTT','GA','TTGGC','AAAAA','CATGT','TGCA','TGGGGG','TCT','GCTTG','TTGA','GTTA','AAAG','TCAA','CATA','CATTA','CTTTT','GCT','CCAATC','TACAT','TCTTC','GTAG','ATGG','TACA','GGTGG','GTAA','GAAT','TACT','ATTGG','ATGTAT','AACT','TATA','TCGA','GAATC','TTAC','ATATC','GAATT','GTTATG','CAAG','GCTT','TCAAA','CTTGT','TAC','AAGCA','GGC','ACCC','ATGTA','ATC','GCC','CTCCCC','TGTG','AGAT','ACGT','TTTGA','TGAAG','TA','CCCCCA','TC','AT','GAT','GGTG','CTAC','GTA','CCAAAA','TCATCA','GAAAG','GCCAA','GATATG','CCATG','CCGG','AGGG','GACT','CCAA','CTC','GGTA','ACTG','AGTC','AGAGAG','AATTC','CAG','ATTAAT','TGCTT','TCTG','ACATG','GGCCA','GGCA','AGAA','ATTC','GAAGA','TGGGG','CCAC','TTTTGG','AAGTC','AGTT','TGGCC','AAACA','CCGA','GGGTA','ATCA','CATT','GCAT','CAGT','TTATG','GGTTTT','CATAA','AAGAA','CATG','GGGT','AAAGA','AGATAT','CAAAG','AATTG','CCCT','TTGGCC','CATAAC','ACAAG','GATAT','GATT','TCTTG','AATGC','CC','GTGT','ACCATG','TATGG','TTGAG','ATATG','GCTTGG','GCATT','TTGCT','TTCTT','TTTGAA','GAAATG','TAT','CTTTG','TACATG','AG','ATGC','TTTCTT','TAGG','CATTG','CATATG','AAAAG','TGGCCA','AGTGGG','AGAC','CTTAA','ATTG','GGG','CTTCA','CCACC','GAG','TGT','CATATC','GGAG','TATATG','GCAC','CACA','TGGTA','TCTCTC','CATGG','ATAC','CTG','TGTTT','ACTTG','GGTAA','CAGA','CCC','GGGG','AGTA','ACT','ATCT','GC','GTTG','CTGG','GACTT','GAGAG','CTTTTT','AGT','TCG','TGAT','TGTAT','TGTGT','GCATC','TGATGC','AAAAAG','AAGT','CTAAG','CATTTC','GCCTT','TTGG','TCCCC','ACAC','GCTAT','CCCCC','TTTT','CAAGA','TCGG','ATAAT','TACCC','CCCC','TGTGA','TCTT','ATATCT','TTTGAG','GAAC','TTTAAA','AAAACC','GCTA','CTTTC','TCCT','CACC','GTGGGA','TAGC','AGC','TTCC','TTTC','GGTAT','CAAC','CATGGT','TGATG','TTCTGG','TGGTAT','CAAGT','GGTTT','TCACA','TGCT','TTCTTC','CTCT','CCAG','GG','GGCT','AGAG','GAAAAA','CT','CAAT','GGCGCC','AGG','CTCC','GGGGG','ATA','TGATGA','ATGGAG','TTACC','ATTA','GGCAT','ATGTTT','CATATA','AGGA','GGT','CAAGC','GCTTC','GAAGT','GATTC','TCATCT','TCA','AGCAA','TACCA','AAGGC','GATCT','CAC','GGAAT','CATCC','CATCCA','AGATGA','AAC','TTG','CCATC','AATT','ATTAT','CCCCA','TGTC','CAAGCA','GCCA','GGTT','AATGG','CTA','GGAGTG','AGCC','AGTTG','TCATT','GGAAG','TAAT','TTACA','GAAAA','AGTGG','ATGTC','GCA','TCTCT','GAATTG','TTCA','ACC','CCATA','CCAAG','AGGTGG','CCATGC','AAACAT','CTGA','AAGCT','GGTAGT','TTAAG','GTGA','TTTCC','TAAG','CCCACT','AGAGA','CAA','TCAACA','GT','TTTTC','ATAGG','GACCA','ATGGTA','TTTTT','CAACT','AATCA','TAATGG','AGGC','GGGC','TTAT','CTTCC','GTCT','GTTAT','GATGC','GAAA','TCCTC','TTAAA','TGTGAT','ACACA','TAGA','CCTTC','CATCA','CCAGAA','CGAA','CGA','CTATG','TATTGG','CATGTA','GTGG','CAT','CCCTT','AATTAA','GATATC','ATTCC','ACTT','TCAG','TTTAA','AATCT','AACC','TGGG','AGATC','ATGCC','GAGGA','CCT','CACTG','ATACAT','AGAAA','CCTTTT','CTCCAT','CCATGG','TT','CTATAG','TGAAGA','CAATG','AAGAAA','ACATGT','TGAC','TTTTTC','AAAA','TCTAG','AAGGG','TGGAT','TGGAA','ACTTC','GGAAA','GTTT','GCCT','ATATAT','CGCCG','CTCCA','TGGC','TTGGG','CTGT','GTCC','GTACC','TCTCA','TGGTC','ATACA','AAGA','GAGGGG','CCTTTG','CCTTA','TCTTTT','CTTCTT','ACTAC','ACATAT','GCAAC','TTTTTG','TCATAA','CTAGG','GAC','ATCAA','AAATGG','CACAA','ATTAA','TTCCC','AC','GCAA','CATAT','TTGATG','CTCTC','AGGAG','TAATG','AAGAG','CAGG','TGTAC','GTC','GATAA','CCTAC','CAATT','AAAAGA','CATTAA','GTTTTT','TATATA','CGTG','TGTTGA','TGCAT','TGAGA','AGCA','GCCC','TAG','TCAC','AAGGT','GGA','CTTA','CCCTCA','TTGC','AAAAGG','CAAAAA','AAATTT','CTTCT','TTA','ATAA','CCACTC','TGA','GTGGGG','GTAT','GGCTT','CCAATA','CCCCTC','CTTCAT','GTTC','AAATCT']

	if id < remain:
		init = id * (seqs_per_procs + 1)
		end = init + seqs_per_procs + 1
	else:
		init = id * seqs_per_procs + remain
		end = init + seqs_per_procs

	totalSeqProc = 0
	if end < numSeqs:
		totalSeqProc = end - init
	else:
		totalSeqProc = numSeqs - init

	rep2d = np.zeros((totalSeqProc, len(langu), len(langu)), dtype=np.int16)
	posSeq = 0
	labels = np.zeros((totalSeqProc, 1), dtype=np.int8)
	print("running in process "+str(id) + " init="+str(init)+" end="+str(end))

	while init < end and init < numSeqs:
		seq = str(TEseqs[init])
		posNucl = 0

		order = -1
		if str(TEids[init]).upper().find("_NEGATIVA") != -1:
			order = 0
		elif str(TEids[init]).upper().find("ALE-") != -1 or str(TEids[init]).upper().find("RETROFIT-") != -1:
			order = 1
		# elif str(TEids[init]).upper().find("ALESIA-") != -1:
		# 	order = 2
		elif str(TEids[init]).upper().find("ANGELA-") != -1:
			order = 3
		elif str(TEids[init]).upper().find("BIANCA-") != -1:
			order = 4
		# elif str(TEids[init]).upper().find("BRYCO-") != -1:
		# 	order = 5
		# elif str(TEids[init]).upper().find("LYCO-") != -1:
		# 	order = 6
		# elif str(TEids[init]).upper().find("GYMCO-") != -1:
		# 	order = 7
		elif str(TEids[init]).upper().find("IKEROS-") != -1:
			order = 8
		elif str(TEids[init]).upper().find("IVANA-") != -1 or str(TEids[init]).upper().find("ORYCO-") != -1:
			order = 9
		# elif str(TEids[init]).upper().find("OSSER-") != -1:
		# 	order = 10
		# elif str(TEids[init]).upper().find("TAR-") != -1:
		# 	order = 11
		elif str(TEids[init]).upper().find("TORK-") != -1:
			order = 12
		elif str(TEids[init]).upper().find("SIRE-") != -1:
			order = 13
		elif str(TEids[init]).upper().find("CRM-") != -1:
			order = 14
		# elif str(TEids[init]).upper().find("CHLAMYVIR-") != -1:
		# 	order = 15
		elif str(TEids[init]).upper().find("GALADRIEL-") != -1:
			order = 16
		elif str(TEids[init]).upper().find("REINA-") != -1:
			order = 17
		elif str(TEids[init]).upper().find("TEKAY-") != -1 or str(TEids[init]).upper().find("DEL-") != -1:
			order = 18
		elif str(TEids[init]).upper().find("ATHILA-") != -1:
			order = 19
		elif str(TEids[init]).upper().find("TAT-") != -1:
			order = 20
		elif str(TEids[init]).upper().find("OGRE-") != -1:
			order = 21
		elif str(TEids[init]).upper().find("RETAND-") != -1:
			order = 22
		# elif str(TEids[init]).upper().find("PHYGY-") != -1:
		# 	order = 23
		# elif str(TEids[init]).upper().find("SELGY-") != -1:
		# 	order = 24
		if order != -1:	
			slide = int(len(seq)/len(langu))
			for i in range(len(langu)):
				for j in range(len(langu)):
					subseq = seq[j*slide:j*slide + slide]
					count = len([x for x in range(len(subseq) - len(langu[i])) if subseq[x:x+len(langu[i])].upper() == langu[i]])
					rep2d[posSeq][i][j] = count

			labels[posSeq][0] = order
		else:
			print("---------- error: --------------")
			print(TEids[init])
			print("--------------------------------")

		if posSeq % 100 == 0:
			print("Process "+str(id)+" doing "+str(posSeq) +" of "+str(totalSeqProc))
		posSeq += 1
		init += 1


	save(file+'.'+multiprocessing.current_process().name, rep2d)
	save(file+'.'+multiprocessing.current_process().name+'_labels.npy', labels)
	rep2d = None
	labels = None
	print("Process "+str(id)+" done!!! ")


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
	threads = int(sys.argv[2])
	filter(seqfile)
	maxLen, numSeqs = maxLength(seqfile+".filtered")
	print(maxLen)
	print("Doing conversion into 2D of "+seqfile+" using Simon method")
	start_time = time.time()
	TEids = []
	TEseqs = []
	for te in SeqIO.parse(seqfile, "fasta"):
		TEids.append(te.id)
		TEseqs.append(te.seq)
	seqs_per_procs = int(numSeqs/threads)+1
	remain = numSeqs % threads
	processes = [multiprocessing.Process(target=positional_kmers, args=[seqfile+".filtered", maxLen, seqs_per_procs, x, remain, TEids, TEseqs, numSeqs]) for x in range(threads)]
	[process.start() for process in processes]
	[process.join() for process in processes]
	print("Joining all partial results..")

	""" This bloc does not run because RAM issues (>126 Gb for 67k sequences)
	finalFile = np.load(seqfile+'.filtered.Process-'+str(1)+'.npy', 'r').astype(dtype=np.int11)
	finalFileLabels = np.load(seqfile+'.filtered.Process-'+str(1)+'_labels.npy', 'r').astype(dtype=np.int8)
	os.remove(seqfile+'.filtered.Process-'+str(1)+'.npy')
	os.remove(seqfile+'.filtered.Process-'+str(1)+'_labels.npy')

	for i in range(2, threads+1):
		filei = np.load(seqfile+'.filtered.Process-'+str(i)+'.npy', 'r').astype(dtype=np.int11)
		finalFile = np.concatenate((finalFile, filei))
		os.remove(seqfile+'.filtered.Process-'+str(i)+'.npy')

		filei = np.load(seqfile+'.filtered.Process-'+str(i)+'_labels.npy', 'r').astype(dtype=np.int8)
		finalFileLabels = np.concatenate((finalFileLabels, filei))
		os.remove(seqfile+'.filtered.Process-'+str(i)+'_labels.npy')"""
	#numSeqs = 67305

	finalFile = np.zeros((numSeqs, 508, 508), dtype=np.int16)
	finalFileLabels = np.zeros((numSeqs, 1), dtype=np.int8)
	index = 0
	index_labels = 0
	for i in range(1, threads+1):
		filei = np.load(seqfile+'.filtered.Process-'+str(i)+'.npy', 'r').astype(dtype=np.int16)
		for j in range(filei.shape[0]):
			if index < numSeqs:
				finalFile[index] = filei[j]
				index += 1
		print(index)
		#os.remove(seqfile+'.filtered.Process-'+str(i)+'.npy')

		filei = np.load(seqfile+'.filtered.Process-'+str(i)+'_labels.npy', 'r').astype(dtype=np.int8)
		for j in range(filei.shape[0]):
			if index_labels < numSeqs:
				finalFileLabels[index_labels] = filei[j]
				index_labels += 1
		print(index_labels)
		#os.remove(seqfile+'.filtered.Process-'+str(i)+'_labels.npy')

	print("saving features file...")
	save(seqfile+'.filtered_PosKmers.npy', finalFile)
	print("done!!")
	print("saving labels file...")
	save(seqfile+'.filtered_PosKmers_labels.npy', finalFileLabels)
	print("done!!")

	end_time = time.time()
	print("Threads time=", end_time - start_time)
