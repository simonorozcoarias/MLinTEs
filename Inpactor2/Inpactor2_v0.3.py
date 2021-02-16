#!/bin/env pyhton

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

# for working in Nvidia RTX 2080 super
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


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
Returns a sliding window (of width n) over data from the iterable"
s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
"""
def window(seq, n):
	it = iter(seq)
	result = tuple(islice(it, n))
	if len(result) == n:
		return result    
	for elem in it:
		result = result[1:] + (elem,)
	return result


"""
This function take a fasta file and write another fasta file with sequences
of length l with a sliding window size of w. Each sequence has the following
formart in the ID: >seq-name_init_end
"""
def splitting_fasta_file(file, length, window, outputDir):
	f = open(outputDir+'/'+file+'_splitted.tmp', 'w')
	for te in SeqIO.parse(outputDir+'/'+file, "fasta"):
		if len(te.seq) <= length:
			f.write('>'+str(te.id).replace('#','_')+'#1#'+str(len(te.seq))+'\n'+str(te.seq)+'\n')
		else:
			for x in range(0, len(te.seq) - length, window):
				f.write('>'+str(te.id).replace('#','_')+'#'+str(x+1)+'#'+str(x+length+1)+'\n'+str(te.seq)+'\n')
	f.close()

"""
This function calculates k-mer frequencies in parallel mode.
"""
def kmer_counting(file, id, seqs_per_procs, kmers, TEids, TEseqs, n, remain, outputDir):
	if id < remain:
		init = id * (seqs_per_procs + 1)
		end = init + seqs_per_procs + 1
	else:
		init = id * seqs_per_procs + remain
		end = init + seqs_per_procs
	#print("running in process "+str(id) + " init="+str(init)+" end="+str(end))
	resultFile = open(outputDir+'/'+file+'.'+multiprocessing.current_process().name, 'w')

	while init < end and init < n:
		frequencies = [0 for x in range(len(kmers))]
		for i in range(len(TEseqs[init])):
			for l in range(1, 7):
				if i+l < len(TEseqs[init]):
					if TEseqs[init][i:i+l].upper().find("N") == -1 and TEseqs[init][i:i+l].upper() in kmers:
						index = kmers.index(TEseqs[init][i:i+l].upper())
						frequencies[index] += 1
		# print (TEids[init])
		frequenciesStrings = [str(x) for x in frequencies]
		resultFile.write(','.join(frequenciesStrings)+'\n')
		# resultMap[TEids[init]] = str(order)+','+','.join(frequenciesStrings)
		init += 1
	resultFile.close()
	#print("Process done in "+str(id))


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
def joining_predictions(seqIds,  predictions, outputDir):
	currentSeq = seqIds[0].split("#")[0]
	currentPred = predictions[0]
	init = seqIds[0].split("#")[1]
	end = seqIds[0].split("#")[2]
	finalIds = []
	for i in range(1, len(seqIds)):
		fields = seqIds[i].split("#")
		if currentSeq == fields[0] and currentPred == predictions[i]: # join predictions
			end = fields[2]
		else: # We do not join predictions
			finalIds.append(currentSeq+"#"+init+"#"+end+"#"+currentPred)
			currentSeq = fields[0]
			init = fields[1]
			end = fields[2]
			currentPred = predictions[i]

	# to save the last sequence
	finalIds.append(currentSeq+"#"+init+"#"+end+"#"+currentPred)

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
		idSeqHost = p.split('#')[0]
		posIni = int(p.split('#')[1])
		posEnd = int(p.split('#')[2])
		lineage = p.split('#')[3]
		seqHost = [str(x.seq) for x in SeqIO.parse(file, 'fasta')]
		outputFile.write('>'+lineage+'_'+idSeqHost+'\n'+seqHost[0][posIni:posEnd]+'\n')
	outputFile.close()


if __name__ == '__main__':
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
		print('FATAL ERROR: Missing fasta file parameter (-f or --file). Exiting')
		sys.exit(0)
	elif os.path.exists(file) == False:
		print('FATAL ERROR: fasta file did not found at path: '+file)
		sys.exit(0)
	if outputDir == None:
		outputDir = os.path.dirname(os.path.realpath(__file__))
		print("WARNING: Missing output directory, using by default: "+outputDir)
	elif os.path.exists(outputDir) == False:
			print('FATAL ERROR: output directory did not found at path: '+file)
			sys.exit(0)
	if threads == None or threads == -1:
		threads = int(psutil.cpu_count())
		print("WARNING: Missing threads parameter, using by default: "+str(threads))
	else:
		threads = int(threads)
	if length == None:
		length = 10000
	else:
		length = int(length)
	if window == None:
		window = 500
	elif window < 6:
		print('FATAL ERROR: window size must be greater than 6. Exiting')
		sys.exit(0)
	else:
		window = int(window)

	if mode == None:
		mode = 'slow'
		print("WARNING: Missing mode parameter, using by default: "+mode)
	if mode not in ['slow', 'fast']:
		print("FATAL ERROR: Mode bad specified: '"+mode+"', you must choose between slow or fast (lower case)")
		sys.exit(0)

	if kmer_file != None:
		if os.path.exists(kmer_file) == False:
			print("FATAL ERROR: K-mer file did not found at the path: '"+kmer_file+"'")
			sys.exit(0)

		kmer_dataframe = pd.read_csv(kmer_file)
		if kmer_dataframe.shape[1] == 508 and mode == 'slow':
			print("FATAL ERROR: K-mer file and mode parameter incopompatibles, your k-mer file was calculate using fast mode and you indicated slow mode to predict LTR retrotransposons")
			sys.exit(0)
		elif kmer_dataframe.shape[1] == 5460 and mode == 'fast':
			print("FATAL ERROR: K-mer file and mode parameter incopompatibles, your k-mer file was calculate using slow mode and you indicated fast mode to predict LTR retrotransposons")
			sys.exit(0)
		kmer_dataframe = None


	##################################################################################
	### First step: split scaffolds into sequences of length l and with a windows w
	print('INFO: Starting to split chromosomes into sequences of length '+str(length)+' with sliding window size '+str(window)+'...')
	start = time.time()	
	splitting_fasta_file(file, length, window, outputDir)
	finish = time.time()
	print('INFO: Splitting done!!!! [time='+str(finish - start)+']')

	##################################################################################
	### Second step: extract k-mer frequencies (1<=k<=6) from sequences (slow mode), or using the 508 most importance k-mers (fast mode)
	TEids = []
	TEseqs = []
	n = 0
	for te in SeqIO.parse(outputDir+'/'+file+'_splitted.tmp', "fasta"):
		TEids.append(te.id)
		TEseqs.append(te.seq)
		n += 1

	if kmer_file == None:
		print('INFO: Starting to extract k-mer frequencies as features from sequences in mode '+mode+'...')
		start = time.time()
		if mode == 'fast':
			# 508 k-mers with most importance score (Orozco-Arias, S., et al. 2021, K-mer-based Machine Learning method to detect and classify LTR-retrotransposons in plant genomes
			kmers = ['A','T','AAAAAA','ATAT','AGGGGG','CCCCCT','TTTTTT','AGCT','GATC','GATGA','ACATA','TATGT','CTTC','CCTT','CTTG','G','ATGT','ACAT','TCT','TTAAT','GA','AAAG','CTAC','CCAT','TATA','CTAG','TAAC','GGGGAG','CATC','GAA','CTT','GTTA','TGCC','TCTC','CTTTTT','AATC','CCAAT','TATAT','CCA','GTAC','TCATC','AGA','CT','GGGGGA','GTA','ATATA','C','ATGC','CTTT','AAAAA','TGGGG','AAGG','GGCC','AAG','GATG','AAAGG','GAAG','TC','GC','TGTA','TTGA','CTG','ATC','TAC','GGC','CATG','TTC','CAAG','AAAAAG','CAG','TAGG','AATGC','CTTGT','CCGG','TATG','GAATT','CCAA','CCTTT','TCCCCC','TACAT','ATATC','AT','GAGA','TTAA','GTAG','TACC','TCAA','TCTG','TGG','AGTT','GTAA','TGC','GGG','TTTTT','TGGGGG','GATAT','TAAG','AAGC','CATTA','CTGG','GAAGA','ACGT','AAGTC','AACTC','GGTGG','CATGT','CCCC','GCAT','CCAATC','TCTTC','GTGT','TTTT','CCC','CAAGT','TGCA','ATGTA','GGAAA','TTAC','TTGGC','TTTGA','TACA','AATTC','GATTGG','ATGG','AACT','GAT','TCGA','GAG','TAG','ACTTG','AG','CTCCCC','CAACT','TTCTT','TCTTG','CAGT','TGGA','TAGC','ATTGG','ATCT','GAGTT','TGTTT','GCTT','GCC','GAGC','GGTTTT','ATTAAT','GTTATG','TCATCA','GATT','CTTTT','AAAACC','AAGCA','GAAAG','ACCC','GAAGT','CATAA','GATATC','TGGTAT','GGTA','TTCT','ATGCC','GGGT','AGTGGG','ATA','GCATT','CCATG','TTGGG','CACA','ATG','CCCCCA','TTATG','GCCAA','CAAAG','CTCT','TGATGA','CCACC','GAATC','TCTT','ATATCT','CCAC','CATAAC','GGTATC','GCT','TTGC','ACAG','GACTT','AGAT','TACATG','TCG','AAGAA','ATGTAT','TTGAG','GACT','CCCACT','AAAA','GGAA','CTAAG','AAACA','ATGTC','TGGCC','CATCT','AGTC','TCACA','ATGAT','ACTG','CC','CATAT','AAAGA','CCTTA','TATGG','CCAG','CTTGA','CATGG','GGGTA','AATCA','CATA','GCTTC','ATTAA','CCCCA','GGCA','TTAT','ACT','TCCC','GAAATG','ATTC','GGCCA','TGAAG','CTATG','AAGGG','AC','AGC','CCATA','AGGG','GCTTG','AAGAAG','ATCA','TA','GAAT','CACC','TCAG','GGGG','ATGATG','AAGT','AGAC','CAGA','AGT','CCGA','CCTA','CTGA','ATATG','GGAT','CAAT','CTTTG','CAAAAA','TCGG','ACAC','GATATG','AGTA','ATCGA','TGGAA','CTCTG','TACT','ATTAT','TTGGCC','GG','AGTGG','TGCAT','AATG','CCAAG','CTC','TGCTT','AGAG','GAAAA','GATGG','TTTCTT','CCCCC','TGTGT','AAC','CCTG','TCCT','TCAAA','CCCT','ATAC','TCA','AAAAG','GCG','GGGGG','TGGCCA','CAAC','AAAAGA','GCATC','CGA','GAAC','AAGGC','TTTTTG','TTGTA','AGAA','GATGAA','TGGAT','GGCGCC','ACAAG','GGTG','GATGC','TTTTGG','CTTAA','ACATG','ACCATG','GGTAT','ACTC','TCAAG','TAGGA','CTCCAT','TAAT','CAA','AACTTT','TTACC','ATTG','GGAGT','ATGTTT','GATTA','TTGCT','AGATAT','GAGG','CCACT','TGT','CTCC','CCT','CAC','GATCT','CCCAA','TGATG','TGAGA','TGTAC','TGTG','GGAAG','GTTC','GGCT','CTCAA','CCAGAA','CTTCTT','CCCTT','GTCT','ACAAA','AAAC','TGAGT','GGT','AGCTT','GCAC','TGGG','CCTAG','GAAAAA','AGCAA','ATAAT','CTTTC','AGATC','GAGGG','GATTC','CAGC','ATATAT','GGA','TTG','GGTT','ACG','GCTA','TGAT','CTTCT','AGCC','TCCCC','GTG','GTC','TGTAA','AGGA','GTGTG','TGGAG','GTGGGA','ACATGT','GT','TTCAT','TTGTT','CATATG','TGGC','GGTAA','AGTTG','CCATGG','GAC','TGGTA','AGAGA','TTCTGG','TTGG','TAATG','TACCC','GGGC','AATGG','CACG','GTT','TTTC','CTTCA','TT','GTGGGG','ACTCC','CTA','GGAG','TGTGA','ATGGTA','ACTT','GTGG','TGA','CATTG','CCCTCC','TTTAA','AGG','GAAA','GCAG','AGAGAG','TGCTCA','GATAAT','CTTCTG','TTGCC','CAGAG','AGCA','CCAAAA','ACC','TGATGC','CTGT','GCA','CAAGC','GTATCA','TCTCA','CGCCG','CATATC','GCCA','CATCC','CAAGCT','ATTTC','CACTG','AATTAA','ATGCAT','ATCAGA','TCGT','TCCAC','ACCACA','CCTTC','GCTTGG','ATAA','GTTT','AGGC','AATCAT','ACTGA','CTTCC','CATGGT','GGAGG','GCTTT','TGAAGA','AACTG','AACAT','GACA','TTTGAA','AAGA','TTCCA','TTCC','AAATCT','GGAC','ATGAA','TGCTG','TCTA','CAGG','CATT','ACA','GCTTTT','GGAGTG','TTAG','TTACA','GTTAT','GCAAC','CCATT','ACACA','CAAGCA','AAGCT','TCTCT','GTCAG','CTCCA','ACTAC','TTGGAG','CAGTT','CGTA','TCTTT','CAAGA','CATAG','AAGAAA','GATGAT','GAAGC','TAGA','CTGAA','AAAAAC','ATACAT','GGTTC','TATATA','GGTTT','CAT','CTCTC','GTGC','ACCG','CACT','CATCCA','TTTTG','TTGTC','GATCTA']
		else:
			kmers = []
			for k in range(1,7):
				for item in itertools.product('ACGT', repeat=k):
					kmers.append(''.join(item))
		seqs_per_procs = int(n/threads)+1
		remain = n % threads
		processes = [multiprocessing.Process(target=kmer_counting, args=[file, x, seqs_per_procs, kmers, TEids, TEseqs, n, remain, outputDir]) for x in range(threads)]
		[process.start() for process in processes]
		[process.join() for process in processes]

		finalFile = open(outputDir+'/'+file+'.kmers', 'w')
		finalFile.write(','.join(kmers)+'\n')
		for i in range(1, threads+1):
			filei = open(outputDir+'/'+file+'.Process-'+str(i), 'r')
			lines = filei.readlines()
			for line in lines:
				finalFile.write(line)
			filei.close()
			os.remove(outputDir+'/'+file+'.Process-'+str(i))
		finalFile.close()
		finish = time.time()
		print('INFO: Extraction of k-mer frequencies done!!!! [time='+str(finish - start)+']')
		kmer_file = outputDir+'/'+file+'.kmers'
	else:
		print('INFO: Using k-mer frequencies from: '+kmer_file+'...')


	##################################################################################
	### Third step: Predict the lineage from the pre-trained DNN.
	print('INFO: Starting to predict the lineages from sequences...')
	start = time.time()	
	predictions = predict_lineages(kmer_file, mode)
	finish = time.time()
	print('INFO: Prediction of the lineages from sequences done!!! [time='+str(finish - start)+']')

	##################################################################################
	### Fourth step: Join equals prediction that are next to 
	print('INFO: Starting join predictions...')
	start = time.time()	
	finalIds = joining_predictions(TEids, predictions, outputDir)
	finish = time.time()
	print('INFO: Joining predictions done!!! [time='+str(finish - start)+']')

	##################################################################################
	### fiveth step: Creating fasta file with LTR retrotransposons classified 
	print('INFO: Creating output fasta file...')
	start = time.time()	
	create_fasta_file(file, finalIds, outputDir)
	finish = time.time()
	print('INFO: Fasta file created!!! [time='+str(finish - start)+']')

	##################################################################################
	### Removing temporal files	
	os.remove(outputDir+'/'+file+'_splitted.tmp')
