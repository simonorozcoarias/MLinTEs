#!/bin/env python

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import Bio
from Bio import SeqIO
import itertools
import time
import multiprocessing
import argparse
import psutil
from joblib import dump, load
import tensorflow as tf
from numpy import argmax
import pandas as pd
from itertools import islice
import numpy as np
from numpy import save
from mpi4py import MPI
import pickle
from tensorflow.keras import backend as K

# for working in Nvidia RTX 2080 super
"""from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)"""


"""
These functions are used to calculated perfomance metrics
"""
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

"""
Sending MPI messages
"""
def receive_mpi_msg(src=MPI.ANY_SOURCE, t=MPI.ANY_TAG, deserialize=False):
	data_dict = {}
	status = MPI.Status()
	data = comm.recv(source=src, tag=t, status=status)
	if deserialize: 
		data = pickle.loads(data)
	data_dict['data'] = data
	data_dict['sender'] = int(status.Get_source())
	return data_dict

"""
Sending MPI messages
"""
def send_mpi_msg(destination, data, serialize=False):
	if serialize: 
		data = pickle.dumps(data)
	comm.send(data, dest=destination)

"""
This function take a fasta file and returns a dict wwith sequences
of length w with a sliding window size of w. Each sequence has the following
formart in the ID: seq-name#init#end
"""
def splitting_fasta_file(file, window):
	dicc_splitted_seqs = {}
	dicc_seq_sizes = {}
	posIni = -1
	posEnd = -1
	for te in SeqIO.parse(file, "fasta"):
		posIni = posEnd + 1
		if len(te.seq) <= window:
			idSeq = str(te.id).replace('#','_')+'#0#'+str(len(te.seq))
			sequence = str(te.seq)
			dicc_splitted_seqs[idSeq] = sequence
			posEnd += 1
		else:
			for x in range(0, len(te.seq), window):
				if (x + window) >= len(te.seq):
					currentEnd = len(te.seq)
				else:
					currentEnd = x + window
				idSeq = str(te.id).replace('#','_')+'#'+str(x)+'#'+str(currentEnd)
				sequence = str(te.seq)[x: currentEnd]
				dicc_splitted_seqs[idSeq] = sequence
				posEnd += 1
		dicc_seq_sizes[te.id] = [posIni, posEnd]
	return dicc_splitted_seqs, dicc_seq_sizes

"""
This function calculates k-mer frequencies in parallel mode.
"""
def kmer_counting(id, seqs_per_procs, kmers, dicc_splitted_seqs, n, remain, outputDir):
	if id < remain:
		init = id * (seqs_per_procs + 1)
		end = init + seqs_per_procs + 1
	else:
		init = id * seqs_per_procs + remain
		end = init + seqs_per_procs

	local_file = open(outputDir+'/file_kmers_'+str(id)+'.csv', 'w')
	while init < end and init < n:
		frequencies = [0 for x in range(len(kmers))]

		# to obtain the real sequence id from dict
		fields = list(dicc_splitted_seqs)[init]
		# to extract the nucleotide sequence form dict
		TEseq = dicc_splitted_seqs[fields]
		#####################

		for i in range(len(TEseq)):
			for l in range(1, 7):
				if i+l < len(TEseq):
					if TEseq[i:i+l].upper().find("N") == -1 and TEseq[i:i+l].upper() in kmers:
						index = kmers.index(TEseq[i:i+l].upper())
						frequencies[index] += 1
		local_file.write(','.join([str(x) for x in frequencies])+'\n')
		init += 1

"""
This function takes the k-mer table pre-calculated and sums them in windows of length l, and write it into a file
"""
def sum_kmers_subseqs(kmers, kmer_table, dicc_seq_sizes, w, l, outputFile, file):
	step = int(l/w)
	resultFile = open(outputDir+'/'+file+'.kmers', 'w')
	resultFile.write(','.join(kmers)+'\n')

	for seqs in list(dicc_seq_sizes):
		posIni = dicc_seq_sizes[seqs][0]
		posEnd = dicc_seq_sizes[seqs][1]
		for i in range(posIni, posEnd, 1):
			if (i + step) >= posEnd:
				currentEnd = posEnd
				stillInSeq = 1
			else:
				currentEnd = i + step
				stillInSeq = 0
			kmer_count = list(kmer_table[i: currentEnd+1, :].sum(axis=0))
			resultFile.write(','.join([str(x) for x in kmer_count])+'\n')
			if stillInSeq == 1:
				break
	resultFile.close()

"""
This function take a fasta file and write another fasta file with sequences
of length l with a sliding window size of w. Each sequence has the following
format in the ID: seq-name#init#end
"""
def splitting_length_window(file, length, window):
	splitted_positions = []
	numSeq = 0
	for te in SeqIO.parse(file, "fasta"):
		if len(te.seq) <= length:
			splitted_positions.append(str(numSeq)+"#0#"+str(len(te.seq)))
		else:
			for x in range(0, len(te.seq) - length, window):
				if (x + length) > len(te.seq):
					end = len(te.seq)
				else:
					end = x + length
				splitted_positions.append(str(numSeq)+"#"+str(x)+"#"+str(end))
		numSeq += 1
	return splitted_positions


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def predict_lineages(kmer_file, mode):
	installation_path = os.path.dirname(os.path.realpath(__file__))
	seq_data = pd.read_csv(kmer_file)
	lineages_names_dic = {0: 'Negative', 1: 'ALE/RETROFIT', 3: 'ANGELA', 4: 'BIANCA', 8: 'IKEROS', 9: 'IVANA/ORYCO', 12: 'TORK', 13: 'SIRE', 14: 'CRM', 16: 'GALADRIEL', 17: 'REINA', 18: 'TEKAY/DEL', 19: 'ATHILA', 20: 'TAT'}

	# Scaling
	if mode == 'fast':
		scaling_path = installation_path+'/std_scaler_fast.bin'
	else:
		scaling_path = installation_path+'/std_scaler.bin'
	scaler = load(scaling_path)
	feature_vectors_scaler = scaler.transform(seq_data)

	#PCA
	if mode == 'fast':
		pca_path = installation_path+'/std_pca_fast.bin'
	else:
		pca_path = installation_path+'/std_pca.bin'
	pca = load(pca_path)
	features_pca = pca.transform(feature_vectors_scaler)

	# loading DNN model and predict labels (lineages)
	if mode == 'fast':
		model_path = installation_path+'/saved-model_fast.hdf5'
	else:
		model_path = installation_path+'/saved-model.hdf5'
	model = tf.keras.models.load_model(model_path, custom_objects={'f1_m':f1_m})
	predictions = model.predict(features_pca)
	lineages_ids = [argmax(x) for x in predictions]
	return [lineages_names_dic[x] for x in lineages_ids]

"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def joining_predictions(file, seqIds,  predictions, outputDir, length):
	# to extract the real sequence ID from the fasta file
	posSeq = int(seqIds[0].split("#")[0])
	ids_from_seqs = [seq.id for seq in SeqIO.parse(file, "fasta")]
	currentSeq = ids_from_seqs[posSeq]
	#####################
	currentPred = predictions[0]
	init = seqIds[0].split("#")[1]
	end = seqIds[0].split("#")[2]
	finalIds = []
	for i in range(1, len(seqIds)):
		fields = seqIds[i].split("#")
		posSeq = int(fields[0])
		nextSeq = ids_from_seqs[posSeq]
		if currentSeq == nextSeq and currentPred == predictions[i]: # join predictions
			end = fields[2]
		else: # We do not join predictions
			finalIds.append(currentSeq+"#"+init+"#"+str(end)+"#"+currentPred)
			currentSeq = nextSeq
			init = fields[1]
			end = fields[2]
			currentPred = predictions[i]

	# to save the last sequence
	finalIds.append(currentSeq+"#"+init+"#"+str(end)+"#"+currentPred)

	# to write the results into a bed file
	f = open(outputDir+'/Inpactor2_annotation.bed', 'w')
	for seqid in finalIds:
		f.write(seqid.replace("#", "\t")+'\n')

	f.close()
	return finalIds
		
"""
This function takes the joined prediction and creates a fasta file containing
all LTR retrotransposon's sequences
"""
def create_fasta_file(file, predictions, outputDir):
	outputFile = open(outputDir+'/Inpactor2_library.fasta', 'w')
	for p in predictions:
		if p.split('#')[3] != "Negative":
			idSeqHost = p.split('#')[0]
			posIni = int(p.split('#')[1])
			posEnd = int(p.split('#')[2])
			lineage = p.split('#')[3]
			seqHost = [str(x.seq) for x in SeqIO.parse(file, 'fasta')]
			outputFile.write('>'+lineage+'_'+idSeqHost+'\n'+seqHost[0][posIni:posEnd]+'\n')
	outputFile.close()


if __name__ == '__main__':
	########################################################
	### Global variables
	global comm, rank, threads, rank_msg

	### Parallel process rank assignment
	comm = MPI.COMM_WORLD
	threads = comm.Get_size()
	rank = comm.Get_rank()
	rank_msg = '[Rank '+str(rank)+' msg]'

	if rank == 0:
		print("\n#########################################################################")
		print("#                                                                       #")
		print("# Inpactor2: LTR Retrotransposon annotator using Deep Neural Networks   #")
		print("#                                                                       #")
		print("#########################################################################\n")

	### read parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', required=True, dest='fasta_file', help='Fasta file containing DNA sequences')
	parser.add_argument('-o', '--output-dir', required=False, dest='outputDir',help='Path of the output directory')
	parser.add_argument('-l', '--suquence-length', required=False, dest='length',help='sub-sequence length')
	parser.add_argument('-w', '--window', required=False, dest='window',help='sliding windows size')
	parser.add_argument('-t', '--threads', required=False, dest='threads', help='Number of threads to be used by Inpactor2')
	parser.add_argument('-m', '--mode', required=False, dest='mode', help='mode to calculate k-mer frequencies [fast or slow]')
	parser.add_argument('-k', '--kmer-file', required=False, dest='kmer_file', help='File containing k-mer frequencies pre-calculated')
	parser.add_argument('-v', '--version', action='version', version='%(prog)s v2.3')

	options = parser.parse_args()
	file = options.fasta_file
	outputDir = options.outputDir
	threads = options.threads
	length = options.length
	window = options.window
	mode = options.mode
	kmer_file = options.kmer_file

	##############################################################################
	## Parameters' validation

	if file == None:
		if rank == 0:
			print('FATAL ERROR: Missing fasta file parameter (-f or --file). Exiting')
		sys.exit(0)
	elif os.path.exists(file) == False:
		if rank == 0:
			print('FATAL ERROR: fasta file did not found at path: '+file)
		sys.exit(0)

	if outputDir == None:
		outputDir = os.path.dirname(os.path.realpath(__file__))
		if rank == 0:
			print("WARNING: Missing output directory, using by default: "+outputDir)
	elif os.path.exists(outputDir) == False:
		if rank == 0:
			print('FATAL ERROR: output directory did not found at path: '+file)
		sys.exit(0)

	if threads == None or threads == -1:
		threads = int(psutil.cpu_count())
		if rank == 0:
			print("WARNING: Missing threads parameter, using by default: "+str(threads))
	else:
		threads = int(threads)

	if length == None:
		length = 10000
	else:
		length = int(length)

	if window == None:
		window = 500
	elif int(window) < 6:
		if rank == 0:
			print('FATAL ERROR: window size must be greater than 6. Exiting')
		sys.exit(0)
	else:
		window = int(window)

	if mode == None:
		mode = 'slow'
		if rank == 0:
			print("WARNING: Missing mode parameter, using by default: "+mode)
	if mode not in ['slow', 'fast']:
		if rank == 0:
			print("FATAL ERROR: Mode bad specified: '"+mode+"', you must choose between slow or fast (lower case)")
		sys.exit(0)

	if kmer_file != None:
		if os.path.exists(kmer_file) == False:
			if rank == 0:
				print("FATAL ERROR: K-mer file did not found at the path: '"+kmer_file+"'")
			sys.exit(0)

		kmer_dataframe = pd.read_csv(kmer_file)
		if kmer_dataframe.shape[1] == 508 and mode == 'slow':
			if rank == 0:
				print("FATAL ERROR: K-mer file and mode parameter incopompatibles, your k-mer file was calculate using fast mode and you indicated slow mode to predict LTR retrotransposons")
			sys.exit(0)
		elif kmer_dataframe.shape[1] == 5460 and mode == 'fast':
			if rank == 0:
				print("FATAL ERROR: K-mer file and mode parameter incopompatibles, your k-mer file was calculate using slow mode and you indicated fast mode to predict LTR retrotransposons")
			sys.exit(0)
		kmer_dataframe = None


	##################################################################################
	### First step: split scaffolds into sequences of length l and with a windows w
	if rank == 0:
		print('INFO: Starting to split chromosomes into sequences of length '+str(length)+' with sliding window size '+str(window)+'...')
		start = time.time()	
		dicc_splitted_seqs, dicc_seq_sizes = splitting_fasta_file(file, window)
		finish = time.time()
		print('INFO: Splitting done!!!! [time='+str(finish - start)+']')

	########################################################
	### MPI region

	##################################################################################
	### Second step part one: extract k-mer frequencies (1<=k<=6) from sequences (slow mode), or using the 508 most importance k-mers (fast mode)
	if kmer_file == None:
		# shared variables
		if mode == 'fast':
			# 508 k-mers with best importance score (Orozco-Arias, S., et al. 2021, K-mer-based Machine Learning method to detect and classify LTR-retrotransposons in plant genomes)
			kmers = ['A','T','AAAAAA','ATAT','AGGGGG','CCCCCT','TTTTTT','AGCT','GATC','GATGA','ACATA','TATGT','CTTC','CCTT','CTTG','G','ATGT','ACAT','TCT','TTAAT','GA','AAAG','CTAC','CCAT','TATA','CTAG','TAAC','GGGGAG','CATC','GAA','CTT','GTTA','TGCC','TCTC','CTTTTT','AATC','CCAAT','TATAT','CCA','GTAC','TCATC','AGA','CT','GGGGGA','GTA','ATATA','C','ATGC','CTTT','AAAAA','TGGGG','AAGG','GGCC','AAG','GATG','AAAGG','GAAG','TC','GC','TGTA','TTGA','CTG','ATC','TAC','GGC','CATG','TTC','CAAG','AAAAAG','CAG','TAGG','AATGC','CTTGT','CCGG','TATG','GAATT','CCAA','CCTTT','TCCCCC','TACAT','ATATC','AT','GAGA','TTAA','GTAG','TACC','TCAA','TCTG','TGG','AGTT','GTAA','TGC','GGG','TTTTT','TGGGGG','GATAT','TAAG','AAGC','CATTA','CTGG','GAAGA','ACGT','AAGTC','AACTC','GGTGG','CATGT','CCCC','GCAT','CCAATC','TCTTC','GTGT','TTTT','CCC','CAAGT','TGCA','ATGTA','GGAAA','TTAC','TTGGC','TTTGA','TACA','AATTC','GATTGG','ATGG','AACT','GAT','TCGA','GAG','TAG','ACTTG','AG','CTCCCC','CAACT','TTCTT','TCTTG','CAGT','TGGA','TAGC','ATTGG','ATCT','GAGTT','TGTTT','GCTT','GCC','GAGC','GGTTTT','ATTAAT','GTTATG','TCATCA','GATT','CTTTT','AAAACC','AAGCA','GAAAG','ACCC','GAAGT','CATAA','GATATC','TGGTAT','GGTA','TTCT','ATGCC','GGGT','AGTGGG','ATA','GCATT','CCATG','TTGGG','CACA','ATG','CCCCCA','TTATG','GCCAA','CAAAG','CTCT','TGATGA','CCACC','GAATC','TCTT','ATATCT','CCAC','CATAAC','GGTATC','GCT','TTGC','ACAG','GACTT','AGAT','TACATG','TCG','AAGAA','ATGTAT','TTGAG','GACT','CCCACT','AAAA','GGAA','CTAAG','AAACA','ATGTC','TGGCC','CATCT','AGTC','TCACA','ATGAT','ACTG','CC','CATAT','AAAGA','CCTTA','TATGG','CCAG','CTTGA','CATGG','GGGTA','AATCA','CATA','GCTTC','ATTAA','CCCCA','GGCA','TTAT','ACT','TCCC','GAAATG','ATTC','GGCCA','TGAAG','CTATG','AAGGG','AC','AGC','CCATA','AGGG','GCTTG','AAGAAG','ATCA','TA','GAAT','CACC','TCAG','GGGG','ATGATG','AAGT','AGAC','CAGA','AGT','CCGA','CCTA','CTGA','ATATG','GGAT','CAAT','CTTTG','CAAAAA','TCGG','ACAC','GATATG','AGTA','ATCGA','TGGAA','CTCTG','TACT','ATTAT','TTGGCC','GG','AGTGG','TGCAT','AATG','CCAAG','CTC','TGCTT','AGAG','GAAAA','GATGG','TTTCTT','CCCCC','TGTGT','AAC','CCTG','TCCT','TCAAA','CCCT','ATAC','TCA','AAAAG','GCG','GGGGG','TGGCCA','CAAC','AAAAGA','GCATC','CGA','GAAC','AAGGC','TTTTTG','TTGTA','AGAA','GATGAA','TGGAT','GGCGCC','ACAAG','GGTG','GATGC','TTTTGG','CTTAA','ACATG','ACCATG','GGTAT','ACTC','TCAAG','TAGGA','CTCCAT','TAAT','CAA','AACTTT','TTACC','ATTG','GGAGT','ATGTTT','GATTA','TTGCT','AGATAT','GAGG','CCACT','TGT','CTCC','CCT','CAC','GATCT','CCCAA','TGATG','TGAGA','TGTAC','TGTG','GGAAG','GTTC','GGCT','CTCAA','CCAGAA','CTTCTT','CCCTT','GTCT','ACAAA','AAAC','TGAGT','GGT','AGCTT','GCAC','TGGG','CCTAG','GAAAAA','AGCAA','ATAAT','CTTTC','AGATC','GAGGG','GATTC','CAGC','ATATAT','GGA','TTG','GGTT','ACG','GCTA','TGAT','CTTCT','AGCC','TCCCC','GTG','GTC','TGTAA','AGGA','GTGTG','TGGAG','GTGGGA','ACATGT','GT','TTCAT','TTGTT','CATATG','TGGC','GGTAA','AGTTG','CCATGG','GAC','TGGTA','AGAGA','TTCTGG','TTGG','TAATG','TACCC','GGGC','AATGG','CACG','GTT','TTTC','CTTCA','TT','GTGGGG','ACTCC','CTA','GGAG','TGTGA','ATGGTA','ACTT','GTGG','TGA','CATTG','CCCTCC','TTTAA','AGG','GAAA','GCAG','AGAGAG','TGCTCA','GATAAT','CTTCTG','TTGCC','CAGAG','AGCA','CCAAAA','ACC','TGATGC','CTGT','GCA','CAAGC','GTATCA','TCTCA','CGCCG','CATATC','GCCA','CATCC','CAAGCT','ATTTC','CACTG','AATTAA','ATGCAT','ATCAGA','TCGT','TCCAC','ACCACA','CCTTC','GCTTGG','ATAA','GTTT','AGGC','AATCAT','ACTGA','CTTCC','CATGGT','GGAGG','GCTTT','TGAAGA','AACTG','AACAT','GACA','TTTGAA','AAGA','TTCCA','TTCC','AAATCT','GGAC','ATGAA','TGCTG','TCTA','CAGG','CATT','ACA','GCTTTT','GGAGTG','TTAG','TTACA','GTTAT','GCAAC','CCATT','ACACA','CAAGCA','AAGCT','TCTCT','GTCAG','CTCCA','ACTAC','TTGGAG','CAGTT','CGTA','TCTTT','CAAGA','CATAG','AAGAAA','GATGAT','GAAGC','TAGA','CTGAA','AAAAAC','ATACAT','GGTTC','TATATA','GGTTT','CAT','CTCTC','GTGC','ACCG','CACT','CATCCA','TTTTG','TTGTC','GATCTA']
		else:
			kmers = []
			for k in range(1,7):
				for item in itertools.product('ACGT', repeat=k):
					kmers.append(''.join(item))

		if rank == 0:
			print('INFO: Starting to extract k-mer frequencies as features from sequences in mode '+mode+'...')
			start = time.time()
			for th in range(1, threads):
				send_mpi_msg(th, dicc_splitted_seqs,serialize=True)

			for th in range(1, threads):
				data_dict = receive_mpi_msg(src=th, deserialize=True)

			print("starting to join partial results")
			kmer_table = np.array(pd.read_csv(outputDir+'/file_kmers_'+str(0)+'.csv'))
			print(kmer_table.shape)
			os.remove(outputDir + '/file_kmers_' + str(0) + '.csv')
			for i in range(1, threads-1):
				local_table = np.array(pd.read_csv(outputDir+'/file_kmers_'+str(i)+'.csv'))
				kmer_table = np.concatenate((kmer_table, local_table), axis=0)
				os.remove(outputDir+'/file_kmers_'+str(i)+'.csv')

			finish = time.time()
			print('INFO: Extraction of k-mer frequencies done!!!! [time='+str(finish - start)+']')

			##################################################################################
			### Second step part two: Predict the lineage from the pre-trained DNN.
			print('INFO: Starting to sum the k-mer frequencies of each subsequence...')
			start = time.time()	
			sum_kmers_subseqs(kmers, kmer_table, dicc_seq_sizes, window, length, outputDir, file)
			TEids = splitting_length_window(file, length, window)
			finish = time.time()
			print('INFO: Sum of each subsequence done!!! [time='+str(finish - start)+']')
		else:
			data_dict = receive_mpi_msg(deserialize=True)
			dicc_splitted_seqs = data_dict['data']
			n = len(list(dicc_splitted_seqs))
			seqs_per_procs = int(n/threads-1)+1
			remain = n % (threads-1)

			kmer_counting(rank-1, seqs_per_procs, kmers, dicc_splitted_seqs, n, remain, outputDir)
			send_mpi_msg(0, 1, serialize=True)

		
		kmer_file = outputDir+'/'+file+'.kmers'
	else:
		print('INFO: Using k-mer frequencies from: '+kmer_file+'...')

	########################################################
	### End MPI region

	##################################################################################
	### Third step: Predict the lineage from the pre-trained DNN.
	if rank == 0:
		print('INFO: Starting to predict the lineages from sequences...')
		start = time.time()	
		predictions = predict_lineages(kmer_file, mode)
		finish = time.time()
		print('INFO: Prediction of the lineages from sequences done!!! [time='+str(finish - start)+']')

		##################################################################################
		### Fourth step: Join equals prediction that are next to 
		print('INFO: Starting join predictions...')
		start = time.time()	
		finalIds = joining_predictions(file, TEids, predictions, outputDir, length)
		finish = time.time()
		print('INFO: Joining predictions done!!! [time='+str(finish - start)+']')

		##################################################################################
		### fiveth step: Creating fasta file with LTR retrotransposons classified 
		print('INFO: Creating output fasta file...')
		start = time.time()	
		create_fasta_file(file, finalIds, outputDir)
		finish = time.time()
		print('INFO: Fasta file created!!! [time='+str(finish - start)+']')
