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
def predict_lineages(kmer_file):
	installation_path = os.path.dirname(os.path.realpath(__file__))
	seq_data = pd.read_csv(kmer_file)
	lineages_names_dic = {0: 'Negative', 1: 'ALE/RETROFIT', 3: 'ANGELA', 4: 'BIANCA', 8: 'IKEROS', 9: 'IVANA/ORYCO', 12: 'TORK', 13: 'SIRE', 14: 'CRM', 16: 'GALADRIEL', 17: 'REINA', 18: 'TEKAY/DEL', 19: 'ATHILA', 20: 'TAT'}

	# Scaling
	scaler = load(installation_path+'/std_scaler.bin')
	feature_vectors_scaler = scaler.transform(seq_data)

	#PCA
	pca = load(installation_path+'/std_pca.bin')
	features_pca = pca.transform(feature_vectors_scaler)

	# loading DNN model and predict labels (lineages)
	model = tf.keras.models.load_model(installation_path+'/saved-model.hdf5', custom_objects={'f1_m':f1_m})
	predictions = model.predict(features_pca)
	lineages_ids = [argmax(x) for x in predictions]
	return [lineages_names_dic[x] for x in lineages_ids]



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
	parser.add_argument('-t', '--threads', required=False, dest='threads', help='Number of threads to be used by Inpactor2')
	parser.add_argument('-v', '--version', action='version', version='%(prog)s v2.0')

	options = parser.parse_args()
	file = options.fasta_file
	outputDir = options.outputDir
	threads = options.threads

	if file == None:
		print('FATAL ERROR: Missing fasta file parameter (-f or --file). Exiting')
		sys.exit(0)
	if outputDir == None:
		outputDir = os.path.dirname(os.path.realpath(__file__))
		print("WARNING: Missing output directory, using by default: "+outputDir)
	if threads == None or threads == -1:
		threads = int(psutil.cpu_count())
		print("WARNING: Missing threads parameter, using by default: "+str(threads))
	else:
		threads = int(threads)


	##################################################################################
	### First step: extract k-mer frequencies (1<=k<=6) from sequences
	print('INFO: Starting to extract k-mer frequencies as features from sequences...')
	kmers = []
	for k in range(1,7):
		for item in itertools.product('ACGT', repeat=k):
			kmers.append(''.join(item))
	TEids = []
	TEseqs = []
	n = 0
	for te in SeqIO.parse(file, "fasta"):
		TEids.append(te.id)
		TEseqs.append(te.seq)
		n += 1
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
	print('INFO: Extraction of k-mer frequencies done!!!!')

	##################################################################################
	### Second step: Predict the lineage from the pre-trained DNN.
	print('INFO: Starting to predict the lineages from sequences...')
	predictions = predict_lineages(outputDir+'/'+file+'.kmers')
	for i in range(len(TEids)):
		print("ID: "+TEids[i]+", prediction: "+predictions[i])
	print('INFO: Prediction of the lineages from sequences done!!!')

