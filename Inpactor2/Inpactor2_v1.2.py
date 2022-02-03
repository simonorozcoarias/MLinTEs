#!/bin/env python

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline
import itertools
import time
import multiprocessing
import subprocess
import argparse
import psutil
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras import backend as K
from numpy import argmax
import pandas as pd
import shutil
from itertools import islice, count
import numpy as np
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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def avg_euclidean_distance(Y_real, Y_predict):
    rest = Y_predict - Y_real
    pow_rest = tf.pow(rest, 2)
    sum_reduced = tf.reduce_sum(pow_rest, 1, keepdims=True)
    result_sqrt = tf.sqrt(sum_reduced)
    result = tf.reduce_mean(result_sqrt)
    return result


def avg_length_diff(Y_real, Y_predict):
    total = 0
    for i in range(Y_real.shape[0]):
        real_len = Y_real[i, 1] - Y_real[i, 0]
        pred_len = Y_predict[i, 1] - Y_predict[i, 0]
        total += abs(real_len - pred_len)
    return total / Y_real.shape[0]


def avg_center_point_diff(Y_real, Y_predict):
    total = 0
    for i in range(Y_real.shape[0]):
        real_center = (Y_real[i, 1] + Y_real[i, 0]) / 2
        pred_center = (Y_predict[i, 1] + Y_predict[i, 0]) / 2
        total += abs(real_center - pred_center)
    return total / Y_real.shape[0]


def FPR(Y_real, Y_predict):
    total = 0
    for i in range(Y_real.shape[0]):
        if (Y_real[i, 1] + Y_real[i, 0]) == 0 and (Y_predict[i, 1] + Y_predict[i, 0]) != 0:
            total += 1
    return (total * 100) / Y_real.shape[0]


def create_dataset(seqFile, x, seqs_per_procs, list_ids, n, remain, total_win_len):
    j = 0
    if x < remain:
        init = x * (seqs_per_procs + 1)
        end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
    else:
        init = x * seqs_per_procs + remain
        end = n if init + seqs_per_procs > n else init + seqs_per_procs

    if end > init:
        dataset = np.zeros((end - init, 5, total_win_len), dtype=np.int8)
        currentSeqID = list_ids[init].split("#")[0]
        currentCompleteSeq = [str(x.seq) for x in SeqIO.parse(seqFile, 'fasta') if x.id == currentSeqID][0]
        while init < end:
            if init >= len(list_ids):
                print("process: " + str(x) + ", init: " + str(init))
            seqIni = int(list_ids[init].split("#")[1])
            seqEnd = int(list_ids[init].split("#")[2])
            iterationSeqID = list_ids[init].split("#")[0]
            # to verify if the next sequence is the same as the previous, in order to prevent read unnecessary times
            # the fasta file. Only when the process need to change the sequence.
            if currentSeqID != iterationSeqID:
                currentCompleteSeq = [str(x.seq) for x in SeqIO.parse(seqFile, 'fasta') if x.id == currentSeqID][0]
                currentSeqID = iterationSeqID
            currentSeq = currentCompleteSeq[seqIni:seqEnd]
            dataset[j] = fasta2one_hot(currentSeq, total_win_len)
            j += 1
            init += 1
        return dataset
    else:
        # there is no elements for processing in this thread
        return np.zeros((1), dtype=np.int8)


def get_final_dataset_size(file, total_win_len):
    seqfile = [x for x in SeqIO.parse(file, 'fasta')]
    list_ids_splitted = []
    for i in range(len(seqfile)):
        for j in range(0, len(str(seqfile[i].seq)), total_win_len):
            initial_pos = j
            end_pos = initial_pos + total_win_len
            list_ids_splitted.append(str(seqfile[i].id) + "#" + str(initial_pos) + "#" + str(end_pos))
    return list_ids_splitted


def fasta2one_hot(sequence, total_win_len):
    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    if len(sequence) < total_win_len:
        rest = ['N' for x in range(total_win_len - len(sequence))]
        sequence += ''.join(rest)

    rep2d = np.zeros((1, 5, len(sequence)), dtype=np.int8)

    for nucl in sequence:
        posLang = langu.index(nucl.upper())
        rep2d[0][posLang][posNucl] = 1
        posNucl += 1
    return rep2d


def one_hot2fasta(dataset):
    langu = ['A', 'C', 'G', 'T', 'N']
    fasta_seqs = ""
    for j in range(dataset.shape[1]):
        if sum(dataset[:, j]) > 0:
            pos = argmax(dataset[:, j])
            fasta_seqs += langu[pos]
    return fasta_seqs


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def Inpactor2_Detect(splitted_genome):
    installation_path = os.path.dirname(os.path.realpath(__file__))

    model = tf.keras.models.load_model(installation_path + '/Inpactor_Detect_model.hdf5')
    predictions = model.predict(splitted_genome)
    return predictions


def sequences_extractor(splitted_genome, position_predictions, x, seqs_per_procs, n, remain, outputdir, max_len_threshold, min_len_threshold, list_ids):
    if x < remain:
        i = x * (seqs_per_procs + 1)
        m = n if i + seqs_per_procs + 1 > n else i + seqs_per_procs + 1
    else:
        i = x * seqs_per_procs + remain
        m = n if i + seqs_per_procs > n else i + seqs_per_procs

    extracted_seqs = np.zeros((1, splitted_genome.shape[1], splitted_genome.shape[2]), dtype=np.int8)
    if i < m:
        k = 0  # index of the splitted_genome dataset of this thread
        predicted_ids = []
        while i < m:
            isTE = position_predictions[i]
            if isTE > 0.90:

                #######
                # to get the positions of the element in the window.
                bestCandidates = adjust_seq_positions(splitted_genome[k, :, :], outputdir, x, max_len_threshold, min_len_threshold)

                if len(bestCandidates) > 0:
                    extracted_seqs_i = np.zeros((len(bestCandidates), splitted_genome.shape[1], splitted_genome.shape[2]), dtype=np.int8)
                    j = 0 # index of the extracted_seq_i np array
                    for c in range(len(bestCandidates)):
                        init = bestCandidates[c][0]
                        end = bestCandidates[c][1]
                        extracted_seqs_i[j, :, init:end] = splitted_genome[k, :, init:end]
                        j += 1
                        # to extract the seq ID to save in the new predicted_list
                        seq_id = list_ids[i].split("#")[0]
                        predicted_ids.append(seq_id + "#" + str(init + (i * splitted_genome.shape[2])) + "#" + str(
                                end + (i * splitted_genome.shape[2])))

                    if np.sum(extracted_seqs) == 0:
                        # this threads have not created a extracted_seq dataset yet
                        # print("en el if: "+str(extracted_seqs.shape))
                        extracted_seqs = extracted_seqs_i
                    else:
                        # this thread have already create a extracted_seq, so concatenate the new one
                        # print("en el else: "+str(extracted_seqs.shape))
                        extracted_seqs = np.concatenate((extracted_seqs, extracted_seqs_i), axis=0)
            i += 1
            k += 1
        IDsfile = open(outputdir+'/predicted_ids_'+str(x)+'.txt', 'w')
        for ID in predicted_ids:
            IDsfile.write(ID+'\n')
        IDsfile.close()
        return extracted_seqs
    else:
        # there is no sequences for processing in this thread
        return np.zeros((1, splitted_genome.shape[1], splitted_genome.shape[2]), dtype=np.int8)


def adjust_seq_positions(extracted_seq, outputDir, idProc, max_len_threshold, min_len_threshold):
    seq1file = open(outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta', 'w')
    iterSeq = one_hot2fasta(extracted_seq)
    seq1file.write('>seq'+'\n' + iterSeq+'\n')
    seq1file.close()

    output = NcbiblastnCommandline(query=outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta',
                                   subject=outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta',
                                   evalue=0.0001, num_threads=1, outfmt=6)()[0]
    bestHits = []
    if output != "":
        hits = output.split('\n')

        for hit in hits:
            columns = hit.split('\t')
            # do not consider alignment between sequences and itself
            if len(columns) > 1 and int(columns[3]) < extracted_seq.shape[1]:
                # do not consider repeated hits
                if [int(columns[6]), int(columns[9])] not in bestHits:
                    # only consider alignment following the same directions (LTR are direct repeats)
                    if int(columns[6]) < int(columns[7]) < int(columns[8]) < int(columns[9]):
                        # consider only alignments lengths longer and shorter than given thresholds
                        if min_len_threshold < int(columns[3]) < max_len_threshold:
                            # searching fot TSR (4-6 bp)
                            if int(columns[6])-6 >= 0 and int(columns[9])+6 < extracted_seq.shape[1]:
                                if hamming_distance(extracted_seq[:, int(columns[6])-6:int(columns[6])],
                                                    extracted_seq[:, int(columns[9]):int(columns[9])+6]) <= 2:
                                    bestHits.append([int(columns[6]), int(columns[9])])
    try:
        os.remove(outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta')
    except:
        print("I could not delete the file: " + outputDir + "/splittedChrWindow_" + str(idProc) + ".fasta")
    return bestHits

def hamming_distance(tsr5, tsr3):
    tsr5_seq = one_hot2fasta(tsr5)
    tsr3_seq = one_hot2fasta(tsr3)

    if len(tsr5_seq) != len(tsr3_seq):
        return len(tsr5_seq)
    distance = 0
    for i in range(len(tsr5_seq)):
        if tsr5_seq[i] != tsr3_seq[i]:
            distance += 1
    return distance

def IDs_extractor(extracted_seqs, list_ids, position_predictions):
    predicted_ids = []
    j = 0
    for i in len(list_ids):
        if position_predictions[i] > 0.5:
            init = 0
            end = 0
            for k in range(extracted_seqs.shape[0]):
                while sum(extracted_seqs[k, :, init]):
                    init += 1
                init += 1
                end = init
                while sum(extracted_seqs[k, :, end] != 0):
                    end += 1
        seq_id = list_ids[i].split("#")[0]
        predicted_ids.append(
            seq_id + "#" + str(init + (i * extracted_seqs.shape[2])) + "#" + str(end + (i * extracted_seqs.shape[2])))
    return predicted_ids


def kmer_extractor_model(dataset):
    # to load pre-calculated weights to extract k-mer frequencies
    installation_path = os.path.dirname(os.path.realpath(__file__))
    weights = np.load(installation_path + '/Weights_SL.npy', allow_pickle=True)
    W_1 = weights[0]
    b_1 = weights[1]
    W_2 = weights[2]
    b_2 = weights[3]
    W_3 = weights[4]
    b_3 = weights[5]
    W_4 = weights[6]
    b_4 = weights[7]
    W_5 = weights[8]
    b_5 = weights[9]
    W_6 = weights[10]
    b_6 = weights[11]

    # to define the CNN model
    inputs = tf.keras.Input(shape=(dataset.shape[1], dataset.shape[2], 1), name="input_1")
    layers_1 = tf.keras.layers.Conv2D(4, (5, 1), strides=(1, 1), weights=[W_1, b_1], activation='relu',
                                      use_bias=True, name='k_1')(inputs)
    layers_1 = tf.keras.backend.sum(layers_1, axis=-2)

    layers_2 = tf.keras.layers.Conv2D(16, (5, 2), strides=(1, 1), weights=[W_2, b_2], activation='relu',
                                      use_bias=True, name='k_2')(inputs)
    layers_2 = tf.keras.backend.sum(layers_2, axis=-2)

    layers_3 = tf.keras.layers.Conv2D(64, (5, 3), strides=(1, 1), weights=[W_3, b_3], activation='relu',
                                      use_bias=True, name='k_3')(inputs)
    layers_3 = tf.keras.backend.sum(layers_3, axis=-2)

    layers_4 = tf.keras.layers.Conv2D(256, (5, 4), strides=(1, 1), weights=[W_4, b_4], activation='relu',
                                      use_bias=True, name='k_4')(inputs)
    layers_4 = tf.keras.backend.sum(layers_4, axis=-2)

    layers_5 = tf.keras.layers.Conv2D(1024, (5, 5), strides=(1, 1), weights=[W_5, b_5], activation='relu',
                                      use_bias=True, name='k_5')(inputs)
    layers_5 = tf.keras.backend.sum(layers_5, axis=-2)

    layers_6 = tf.keras.layers.Conv2D(4096, (5, 6), strides=(1, 1), weights=[W_6, b_6], activation='relu',
                                      use_bias=True, name='k_6')(inputs)
    layers_6 = tf.keras.backend.sum(layers_6, axis=-2)

    layers = tf.concat([layers_1, layers_2, layers_3, layers_4, layers_5, layers_6], 2)
    outputs = tf.keras.layers.Flatten()(layers)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    kmers = model.output
    for layer in model.layers:
        layer.trainable = False
    return model


"""
This function calculates k-mer frequencies in parallel mode.
"""
def Inpactor2_kmer(extracted_sequences, batch_size):
    kmer_extractor = kmer_extractor_model(extracted_sequences)
    kmer_counts = kmer_extractor.predict(extracted_sequences, batch_size=batch_size)
    return kmer_counts


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def Inpactor2_Class(seq_data):
    installation_path = os.path.dirname(os.path.realpath(__file__))
    lineages_names_dic = {0: 'Negative', 1: 'ALE/RETROFIT', 3: 'ANGELA', 4: 'BIANCA', 8: 'IKEROS', 9: 'IVANA/ORYCO',
                          11: 'TAR', 12: 'TORK', 13: 'SIRE', 14: 'CRM', 16: 'GALADRIEL', 17: 'REINA', 18: 'TEKAY/DEL',
                          19: 'ATHILA', 20: 'TAT'}

    # Scaling
    scaling_path = installation_path + '/std_scaler.bin'
    scaler = load(scaling_path)
    feature_vectors_scaler = scaler.transform(seq_data)

    # PCA
    pca_path = installation_path + '/std_pca.bin'
    pca = load(pca_path)
    features_pca = pca.transform(feature_vectors_scaler)

    # loading DNN model and predict labels (lineages)
    model_path = installation_path + '/Inpactor_Class.hdf5'
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
    predictions = model.predict(features_pca)
    lineages_ids = [argmax(x) for x in predictions]
    return [lineages_names_dic[x] for x in lineages_ids]


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def create_bed_file(seqIds, predictions, outputDir):
    # to extract the real sequence ID from the fasta file
    finalIds = []
    for i in range(0, len(seqIds)):
        # to save the last sequence
        finalIds.append(seqIds[i] + "#" + predictions[i])

    # to write the results into a bed file
    f = open(outputDir + '/Inpactor2_annotation.bed', 'w')
    for seqid in finalIds:
        f.write(seqid.replace("#", "\t") + '\n')
    f.close()
    return finalIds


"""
This function takes the joined prediction and creates a fasta file containing
all LTR retrotransposon's sequences
"""
def create_fasta_file(predicted_ltr_rts, finalIds, x, seqs_per_procs, n, remain):
    results = ""
    i = 0
    if x < remain:
        init = x * (seqs_per_procs + 1)
        end = init + seqs_per_procs + 1
    else:
        init = x * seqs_per_procs + remain
        end = n if init + seqs_per_procs > n else init + seqs_per_procs
    while init < end:
        p = finalIds[init]
        if p.split('#')[3] != "Negative":
            idSeqHost = p.split('#')[0]
            initPos = p.split('#')[1]
            endpos = p.split('#')[2]
            lineage = p.split('#')[3]
            seqHost = one_hot2fasta(predicted_ltr_rts[i, :, :])
            results += '>' + lineage + '_' + idSeqHost + '_' + initPos + '_' + endpos + '\n' + seqHost + '\n'
        init += 1
        i += 1
    return results


if __name__ == '__main__':
    print("\n#########################################################################")
    print("#                                                                       #")
    print("# Inpactor2: LTR Retrotransposon annotator using Deep Neural Networks   #")
    print("#                                                                       #")
    print("#########################################################################\n")

    ### read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, dest='fasta_file', help='Fasta file containing DNA sequences')
    parser.add_argument('-o', '--output-dir', required=False, dest='outputDir', help='Path of the output directory')
    parser.add_argument('-t', '--threads', required=False, dest='threads',
                        help='Number of threads to be used by Inpactor2')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v2.3')

    options = parser.parse_args()
    file = options.fasta_file
    outputDir = options.outputDir
    threads = options.threads

    ##############################################################################
    # Parameters' validation

    if file is None:
        print('FATAL ERROR: Missing fasta file parameter (-f or --file). Exiting')
        sys.exit(0)
    elif not os.path.exists(file):
        print('FATAL ERROR: fasta file did not found at path: ' + file)
        sys.exit(0)
    if outputDir is None:
        outputDir = os.path.dirname(os.path.realpath(__file__))
        print("WARNING: Missing output directory, using by default: " + outputDir)
    elif not os.path.exists(outputDir):
        print('FATAL ERROR: output directory did not found at path: ' + outputDir)
        sys.exit(0)
    if threads is None or threads == -1:
        threads = int(psutil.cpu_count())
        print("WARNING: Missing threads parameter, using by default: " + str(threads))
    else:
        threads = int(threads)

    ##################################################################################
    # global configuration variables
    total_win_len = 50000
    batch_size = 2
    max_len_threshold = 25000
    min_len_threshold = 2000

    ##################################################################################
    # First step: Split input sequences into chunks of 50k bp and convert it into one-hot coding
    print('INFO: Starting to split input sequences into chunks of size ' + str(
        total_win_len) + ' and converting them into one-hot coding...')
    start = time.time()
    list_ids = get_final_dataset_size(file, total_win_len)

    # Run in parallel the splitter
    n = len(list_ids)
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(create_dataset,
                                     args=[file, x, seqs_per_procs, list_ids, n, remain, total_win_len]) for x in
                                     range(threads)]
    localTables = [p.get() for p in localresults]
    splitted_genome = localTables[0]
    for i in range(1, len(localTables)):
        if len(localTables[i].shape) > 1:
            splitted_genome = np.concatenate((splitted_genome, localTables[i]), axis=0)

    finish = time.time()
    print('INFO: Splitting of input sequences done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Second step: Predict initial and end position of LTR-RTs in each chunk
    print('INFO: Starting to predict which genome chunk contains LTR-RTs...')
    start = time.time()
    position_predictions = Inpactor2_Detect(splitted_genome)
    finish = time.time()
    print('INFO: LTR-RTs containing prediction done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Third step: Extract sequences predicted as LTR-RTs
    print('INFO: Starting to extract sequences predicted as LTR-RTs...')
    start = time.time()

    # Run in parallel the extraction
    n = position_predictions.shape[0]
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    splitted_genome_list = []
    for i in range(threads):
        if i < remain:
            init = i * (seqs_per_procs + 1)
            end = init + seqs_per_procs + 1
        else:
            init = i * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        splitted_genome_list.append(splitted_genome[init:end, :, :])

    splitted_genome = None  # to clean unusable variable
    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(sequences_extractor,
                                args=[splitted_genome_list[x], position_predictions, x, seqs_per_procs, n, remain,
                                outputDir, max_len_threshold, min_len_threshold, list_ids]) for x in range(threads)]
    localTables = [p.get() for p in localresults]

    # to join local results of extracted sequences
    ltr_rts_predicted = localTables[0]
    for i in range(1, len(localTables)):
        if np.sum(localTables[i]) > 0:
            ltr_rts_predicted = np.concatenate((ltr_rts_predicted, localTables[i]), axis=0)

    # to join local results of predicted IDs
    ids_predicted = []
    for i in range(threads):
        IDsfile = open(outputDir + '/predicted_ids_' + str(i) + '.txt', 'r')
        lines = IDsfile.readlines()
        for line in lines:
            ids_predicted.append(line.replace('\n', ''))
        IDsfile.close()
        os.remove(outputDir + '/predicted_ids_' + str(i) + '.txt')

    splitted_genome_list = None  # to clean unusable variable
    position_predictions = None  # to clean unusable variable
    finish = time.time()
    print('INFO: Extraction done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Fourth step: k-mer Counting (1<=k<=6) from sequences using a DNN
    print('INFO: Starting to count k-mer frequencies using a DNN...')
    start = time.time()
    kmer_counts = Inpactor2_kmer(ltr_rts_predicted, batch_size)
    finish = time.time()
    print('INFO: K-mer counting done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Fifth step: Predict the lineage from the pre-trained DNN.
    print('INFO: Starting to predict the lineages from sequences...')
    start = time.time()
    predictions = Inpactor2_Class(kmer_counts)
    kmer_counts = None
    finish = time.time()
    print('INFO: Prediction of the lineages from sequences done!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Sixth step: Join equals prediction that are next to
    print('INFO: Starting join predictions to create the bed output...')
    start = time.time()
    finalIds = create_bed_file(ids_predicted, predictions, outputDir)
    finish = time.time()
    print('INFO: Joining predictions done!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Seventh step: Creating fasta file with LTR retrotransposons classified
    print('INFO: Creating output fasta file...')
    start = time.time()

    # Run in parallel the fasta creation
    n = len(finalIds)
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(create_fasta_file,
                                     args=[ltr_rts_predicted, finalIds, x, seqs_per_procs, n, remain]) for x in
                    range(threads)]
    localSequences = [p.get() for p in localresults]
    outputFile = open(outputDir + '/Inpactor2_library.fasta', 'w')
    for i in range(len(localSequences)):
        outputFile.write(localSequences[i])
    outputFile.close()
    finish = time.time()
    print('INFO: Fasta file created!!! [time=' + str(finish - start) + ']')
