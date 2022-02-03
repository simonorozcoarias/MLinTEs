#!/bin/env python

import sys
import os
from turtle import color

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline
import time
import multiprocessing
import argparse
import psutil
from joblib import dump, load
import tensorflow as tf
from tensorflow.keras import backend as K
from numpy import argmax
import numpy as np
import tracemalloc

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


def check_nucleotides(list_seqs):
    for seq in list_seqs:
        noDNAlanguage = [nucl for nucl in str(seq) if nucl.upper() not in ['A', 'C', 'T', 'G', 'N', '\n']]
        if len(noDNAlanguage) > 0:
            return 1
    return 0


def create_dataset(list_seqs, total_win_len, outputdir, x):
    j = 0
    if len(list_seqs) > 0:
        dataset = np.zeros((len(list_seqs), 5, total_win_len), dtype=np.int8)
        for i in range(len(list_seqs)):
            dataset[j] = fasta2one_hot(list_seqs[i], total_win_len)
            j += 1

        if dataset.shape[1] > 1:  # Process find some LTR-RTs
            np.save(outputdir + '/dataset_2d_' + str(x) + '.npy', dataset.astype(np.int8))
            return np.zeros((10, 10), dtype=np.int8)
        else:  # Process did not find any LTR-RT
            return np.zeros((1, 1), dtype=np.int8)
        return dataset
    else:
        # there is no elements for processing in this thread
        return np.zeros((1), dtype=np.int8)


def get_final_dataset_size(file, total_win_len):
    seqfile = [x for x in SeqIO.parse(file, 'fasta')]
    list_ids_splitted = []
    list_seq_splitter = []
    for i in range(len(seqfile)):
        for j in range(0, len(str(seqfile[i].seq)), total_win_len):
            if "#" in str(seqfile[i].id):
                print("FATAL ERROR: Sequence ID ("+str(seqfile[i].id)+") must no contain character '#', please remove "
                                                                      "all of these and re-run Inpactor2")
                sys.exit(0)
            initial_pos = j
            end_pos = initial_pos + total_win_len
            if end_pos > len(str(seqfile[i].seq)):
                end_pos = len(str(seqfile[i].seq))
            list_ids_splitted.append(str(seqfile[i].id) + "#" + str(initial_pos) + "#" + str(end_pos))
            list_seq_splitter.append(str(seqfile[i].seq)[initial_pos:end_pos])
    return list_ids_splitted, list_seq_splitter


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
    tracemalloc.start()  # to track memory used

    installation_path = os.path.dirname(os.path.realpath(__file__))
    model = tf.keras.models.load_model(installation_path + '/Models/Inpactor_Detect_model.hdf5')

    current, peak = tracemalloc.get_traced_memory()  # to track memory used

    predictions = model.predict(splitted_genome)

    print(
        f"Method: Inpactor2_Detect.\n Current memory usage is {current / 10 ** 9}GB; Peak was {peak / 10 ** 9}GB")  # to track memory used
    tracemalloc.stop()  # to track memory used
    return predictions


def sequences_extractor(splitted_genome, position_predictions, x, seqs_per_procs, n, remain, outputdir,
                        max_len_threshold, min_len_threshold, list_ids, tg_ca, TSD):
    """if x < remain:
        i = x * (seqs_per_procs + 1)
        m = n if i + seqs_per_procs + 1 > n else i + seqs_per_procs + 1
    else:
        i = x * seqs_per_procs + remain
        m = n if i + seqs_per_procs > n else i + seqs_per_procs"""
    i = 0
    m = splitted_genome.shape[0]
    extracted_seqs = np.zeros((1, 1), dtype=np.int8)
    if i < m:
        k = 0  # index of the splitted_genome dataset of this thread
        predicted_ids = []
        while i < m:
            isTE = position_predictions[i, 0]
            if isTE > 0.5:

                #######
                # to get the positions of the element in the window.
                bestCandidates = adjust_seq_positions(splitted_genome[k, :, :], outputdir, x, max_len_threshold,
                                                      min_len_threshold, tg_ca, TSD)

                if len(bestCandidates) > 0:
                    extracted_seqs_i = np.zeros(
                        (len(bestCandidates), splitted_genome.shape[1], splitted_genome.shape[2]), dtype=np.int8)
                    j = 0  # index of the extracted_seq_i np array
                    for c in range(len(bestCandidates)):
                        init = bestCandidates[c][0]
                        end = bestCandidates[c][1]
                        extracted_seqs_i[j, :, init:end] = splitted_genome[k, :, init:end]
                        j += 1
                        # to extract the seq ID to save in the new predicted_list
                        seq_id = list_ids[i].split("#")[0]
                        predicted_ids.append(seq_id + "#" + str(init + (i * splitted_genome.shape[2])) + "#" + str(
                            end + (i * splitted_genome.shape[2])) + "#" + str(isTE))

                    if np.sum(extracted_seqs) == 0:
                        # this threads have not created a extracted_seq dataset yet
                        extracted_seqs = extracted_seqs_i
                    else:
                        # this thread have already create a extracted_seq, so concatenate the new one
                        extracted_seqs = np.concatenate((extracted_seqs, extracted_seqs_i), axis=0)
            i += 1
            k += 1
        IDsfile = open(outputdir + '/predicted_ids_' + str(x) + '.txt', 'w')
        for ID in predicted_ids:
            IDsfile.write(ID + '\n')
        IDsfile.close()

        """if extracted_seqs.shape[1] > 1:  # Process find some LTR-RTs
            np.save(outputdir + '/extracted_seqs_2d_' + str(x) + '.npy', extracted_seqs.astype(np.int8))
            return np.zeros((10, 10), dtype=np.int8)
        else:  # Process did not find any LTR-RT
            return np.zeros((1, 1), dtype=np.int8)"""
        return extracted_seqs
    else:
        # there is no sequences for processing in this thread
        return np.zeros((1, 1), dtype=np.int8)


def adjust_seq_positions(extracted_seq, outputDir, idProc, max_len_threshold, min_len_threshold, tg_ca, TSD):
    seq1file = open(outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta', 'w')
    iterSeq = one_hot2fasta(extracted_seq)
    seqnum = 0
    subseq_len = 50000
    for i in range(0, 50000, subseq_len):
        seq1file.write('>seq_'+str(seqnum) + '\n' + iterSeq[i:i+subseq_len] + '\n')
        seqnum += 1
    seq1file.close()

    try:
        output = NcbiblastnCommandline(query=outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta',
                                       subject=outputDir + '/splittedChrWindow_' + str(idProc) + '.fasta',
                                       evalue=0.0001, num_threads=64, outfmt=6)()[0]
    except Exception as e:
        print("FATAL ERROR. BlastN could not execute the alignment, please re-execute Inpactor2...")
        print(e)
        sys.exit(0)
    bestHits = []
    if output != "":
        hits = output.split('\n')

        for hit in hits:
            columns = hit.split('\t')
            # do not consider alignment between sequences and itself.
            # Only consider hits between chunks in order (from left to right)
            if len(columns) > 1: #and columns[0] != columns[1] and int(columns[0].split("_")[-1]) < int(columns[1].split("_")[-1]):
                ini_factor = int(columns[0].split("_")[-1]) * subseq_len
                end_factor = int(columns[1].split("_")[-1]) * subseq_len
                # do not consider repeated hits
                if [int(columns[6])+ini_factor, int(columns[9])+end_factor] not in bestHits:
                    # only consider alignment following the same directions (LTR are direct repeats)
                    if int(columns[6]) < int(columns[7]) and int(columns[8]) < int(columns[9]):
                        # consider only alignments lengths longer and shorter than given thresholds
                        # consider only alignments larger than 100 pb (candidate LTRs)
                        if min_len_threshold < (int(columns[6])+ini_factor + int(columns[9])+end_factor) < max_len_threshold and int(columns[3]) > 100:
                            # searching fot TSD (5 bp)
                            if int(columns[6]) - 5 >= 0 and int(columns[9]) + 5 < extracted_seq.shape[1]:
                                if hamming_distance(extracted_seq[int(columns[6])+ini_factor - 6:int(columns[6])+ini_factor],
                                                    extracted_seq[int(columns[9])+end_factor:int(columns[9])+end_factor + 6]) <= TSD:
                                    # If TG-CA variable is activated, search if hits starts with TG and ends in CA
                                    if tg_ca:
                                        if iterSeq[int(columns[6])+ini_factor:int(columns[6])+ini_factor + 2] == 'TG' and iterSeq[int(
                                                columns[7])+end_factor - 2:int(columns[7])+end_factor] == 'CA':
                                            bestHits.append([int(columns[6])+ini_factor, int(columns[9])+end_factor])
                                    else:
                                        bestHits.append([int(columns[6])+ini_factor, int(columns[9])+end_factor])
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
    weights = np.load(installation_path + '/Models/Weights_SL.npy', allow_pickle=True)
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
This function uses the FNN to automatically filter non-intact sequences.
"""
def Inpactor2_Filter(kmer_counts, ids_predicted):
    installation_path = os.path.dirname(os.path.realpath(__file__))
    # Scaling
    scaling_path = installation_path + '/Models/std_scaler_filter.bin'
    scaler = load(scaling_path)
    feature_vectors_scaler = scaler.transform(kmer_counts)

    # PCA
    pca_path = installation_path + '/Models/std_pca_filter.bin'
    pca = load(pca_path)
    features_pca = pca.transform(feature_vectors_scaler)

    # loading DNN model and predict labels (lineages)
    model_path = installation_path + '/Models/Inpactor_Filter.hdf5'
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
    predictions = model.predict(features_pca)
    binary_predictions = [argmax(x) for x in predictions]
    filtered_elements = np.zeros((len([x for x in binary_predictions if x == 0]), kmer_counts.shape[1]), dtype=np.int8)
    new_ids_predicted = []
    j = 0  # index of filtered_elements array
    for i in range(len(binary_predictions)):
        if binary_predictions[i] == 0:
            filtered_elements[j, :] = kmer_counts[i, :]
            new_ids_predicted.append(ids_predicted[i] + "#" + str(predictions[i, 0]))
            j += 1
    return filtered_elements, new_ids_predicted


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def Inpactor2_Class(seq_data):
    installation_path = os.path.dirname(os.path.realpath(__file__))
    lineages_names_dic = {0: 'Negative', 1: 'ALE/RETROFIT', 3: 'ANGELA', 4: 'BIANCA', 8: 'IKEROS', 9: 'IVANA/ORYCO',
                          11: 'TAR', 12: 'TORK', 13: 'SIRE', 14: 'CRM', 16: 'GALADRIEL', 17: 'REINA', 18: 'TEKAY/DEL',
                          19: 'ATHILA', 20: 'TAT'}

    # Scaling
    scaling_path = installation_path + '/Models/std_scaler.bin'
    scaler = load(scaling_path)
    feature_vectors_scaler = scaler.transform(seq_data)

    # PCA
    pca_path = installation_path + '/Models/std_pca.bin'
    pca = load(pca_path)
    features_pca = pca.transform(feature_vectors_scaler)

    # loading DNN model and predict labels (lineages)
    model_path = installation_path + '/Models/Inpactor_Class.hdf5'
    model = tf.keras.models.load_model(model_path, custom_objects={'f1_m': f1_m})
    predictions = model.predict(features_pca)
    lineages_ids = [argmax(x) for x in predictions]
    perc_list = []
    for i in range(predictions.shape[0]):
        perc_list.append(predictions[i, lineages_ids[i]])
    return [lineages_names_dic[x] for x in lineages_ids], perc_list


"""
This function predicts the lineage of each sequence in the k-mer file using a pre-trained DNN
"""
def create_bed_file(seqIds, predictions, percentajes, outputDir, curation):
    # to extract the real sequence ID from the fasta file
    finalIds = []
    for i in range(0, len(seqIds)):
        # to save the last sequence
        finalIds.append(seqIds[i] + "#" + predictions[i])

    # to write the results into a bed file
    f = open(outputDir + '/Inpactor2_predictions.tab', 'w')
    i = 0
    for seqid in finalIds:
        columns = seqid.split("#")
        idseq = columns[0]
        initPos = columns[1]
        endPos = columns[2]
        percDect = columns[3]
        perClass = str(percentajes[i])
        if curation:
            percFilt = columns[4]
            lineage = columns[5]
            f.write(idseq + '\t' + initPos + '\t' + endPos + '\t' + str(
                int(endPos) - int(
                    initPos)) + '\t' + lineage + '\t' + percDect + '\t' + percFilt + '\t' + perClass + '\n')
        else:
            lineage = columns[4]
            f.write(idseq + '\t' + initPos + '\t' + endPos + '\t' + str(
                int(endPos) - int(
                    initPos)) + '\t' + lineage + '\t' + percDect + '\t-\t' + perClass + '\n')
        i += 1
    f.close()
    return finalIds


"""
This function takes the joined prediction and creates a fasta file containing
all LTR retrotransposon's sequences
"""
def create_fasta_file(predicted_ltr_rts, finalIds, x, seqs_per_procs, n, remain, outputDir, curation):
    res = ""
    i = 0
    result_file = open(outputDir + '/Inpactor2_library_' + str(x) + '.fasta', 'w')
    if x < remain:
        init = x * (seqs_per_procs + 1)
        end = init + seqs_per_procs + 1
    else:
        init = x * seqs_per_procs + remain
        end = n if init + seqs_per_procs > n else init + seqs_per_procs
    while init < end < n:
        p = finalIds[init]
        columns = p.split("#")
        lineage = ""
        if curation:
            lineage = columns[5]
        else:
            lineage = columns[4]
        if lineage != "Negative":
            idseq = columns[0]
            initPos = columns[1]
            endPos = columns[2]
            seqHost = one_hot2fasta(predicted_ltr_rts[i, :, :])
            results = '>' + idseq + '_' + initPos + '_' + endPos + '#LTR/' + lineage.replace('/',
                                                                                             '-') + '\n' + seqHost + '\n'
            result_file.write(results)
        init += 1
        i += 1
    return res


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
    parser.add_argument('-a', '--annotate', required=False, dest='annotate',
                        help='Annotate LTR retrotransposons? [yes or not]')
    parser.add_argument('-m', '--max-len', required=False, dest='max_len_threshold',
                        help='Maximum length for detecting LTR-retrotransposons')
    parser.add_argument('-n', '--min-len', required=False, dest='min_len_threshold',
                        help='Minimum length for detecting LTR-retrotransposons')
    parser.add_argument('-i', '--tg-ca', required=False, dest='tg_ca',
                        help='Keep only elements with TG-CA-LTRs? [yes or not]')
    parser.add_argument('-d', '--tsd', required=False, dest='TSD',
                        help='Number of mismatches allowed to keep LTR-retrotransposons')
    parser.add_argument('-c', '--curated', required=False, dest='curation',
                        help='keep on only intact elements? [yes or not]')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s v2.3')

    options = parser.parse_args()
    file = options.fasta_file
    outputDir = options.outputDir
    threads = options.threads
    annotate = options.annotate
    max_len_threshold = options.max_len_threshold
    min_len_threshold = options.min_len_threshold
    tg_ca = options.tg_ca
    TSD = options.TSD
    curation = options.curation

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
    if annotate is None:
        annotate = 'yes'
        print("WARNING: Missing annotation parameter (-a or --annotate), using by default: " + str(annotate))
    elif annotate.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -a parameter: ' + annotate + '. This parameter must be yes or no')
        sys.exit(0)
    if max_len_threshold is None:
        max_len_threshold = 28000
        print("WARNING: Missing max length parameter, using by default: 28000")
    else:
        max_len_threshold = int(max_len_threshold)
    if min_len_threshold is None:
        min_len_threshold = 2000
        print("WARNING: Missing min length parameter, using by default: 2000")
    else:
        min_len_threshold = int(min_len_threshold)
    if tg_ca is None:
        tg_ca = False
        print("WARNING: Missing TG-CA filter parameter, using by default: no")
    elif tg_ca.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -i parameter: ' + tg_ca + '. This parameter must be yes or no')
        sys.exit(0)
    else:
        if tg_ca.upper() == 'YES':
            tg_ca = True
        else:
            tg_ca = False
    if TSD is None:
        TSD = 1
        print("WARNING: Missing TSD mismatch number parameter, using by default: 1")
    else:
        TSD = int(TSD)
    if curation is None:
        curation = False
        print("WARNING: Missing curation parameter, using by default: yes")
    elif curation.upper() not in ['YES', 'NO']:
        print('FATAL ERROR: unknown value of -c parameter: ' + curation + '. This parameter must be yes or no')
        sys.exit(0)
    else:
        if curation.upper() == 'YES':
            curation = True
        else:
            curation = False

    ##################################################################################
    # global configuration variables
    total_win_len = 50000
    batch_size = 2
    total_time = []

    ##################################################################################
    # First step: Split input sequences into chunks of 50k bp and convert it into one-hot coding
    tf.keras.backend.clear_session()  # to clean GPU memory
    print('INFO: Splitting input sequences into chunks of size ' + str(
        total_win_len) + ' and converting them into one-hot coding...')
    start = time.time()
    tracemalloc.start()  # to track memory used
    list_ids, list_seqs = get_final_dataset_size(file, total_win_len)
    current, peak = tracemalloc.get_traced_memory()  # to track memory used
    print(f"Method: get_final_dataset_size.\n Current memory usage is {current / 10 ** 9}GB; Peak was {peak / 10 ** 9}GB")  # to track memory used
    tracemalloc.stop()  # to track memory used

    # To validate that sequences only contain valid DNA nucleotides in parallel
    n = len(list_seqs)
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    ini_per_thread = []
    end_per_thread = []
    for p in range(threads):
        if p < remain:
            init = p * (seqs_per_procs + 1)
            end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
        else:
            init = p * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        ini_per_thread.append(init)
        end_per_thread.append(end)
    pool = multiprocessing.Pool(processes=threads)
    tracemalloc.start()  # to track memory used
    localresults = [pool.apply_async(check_nucleotides,
                                     args=[list_seqs[ini_per_thread[x]:end_per_thread[x]]]) for x in range(threads)]
    localChecks = [p.get() for p in localresults]
    for i in range(len(localChecks)):
        if localChecks[i] == 1:
            print("FATAL ERROR: DNA sequences must contain only A, C, G, T, or N characters, please fix it and "
                  "re-run Inpactor2")
            sys.exit(0)

    # Run in parallel the splitter
    n = len(list_ids)
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    ini_per_thread = []
    end_per_thread = []
    for p in range(threads):
        if p < remain:
            init = p * (seqs_per_procs + 1)
            end = n if init + seqs_per_procs + 1 > n else init + seqs_per_procs + 1
        else:
            init = p * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        ini_per_thread.append(init)
        end_per_thread.append(end)
    pool = multiprocessing.Pool(processes=threads)
    tracemalloc.start()  # to track memory used
    localresults = [pool.apply_async(create_dataset,
                                     args=[list_seqs[ini_per_thread[x]:end_per_thread[x]], total_win_len, outputDir, x]) for x in
                    range(threads)]
    localTables = [p.get() for p in localresults]
    list_seqs = None
    current, peak = tracemalloc.get_traced_memory()  # to track memory used
    print(
        f"Method: create_dataset.\n Current memory usage is {current / 10 ** 9}GB; Peak was {peak / 10 ** 9}GB")  # to track memory used
    #print("Local tables size="+str(sys.getsizeof(localTables) / 10 ** 9)+" GB")
    tracemalloc.stop()  # to track memory used
    tracemalloc.start()  # to track memory used
    splitted_genome = np.zeros((n, 5, total_win_len), dtype=np.int8)
    print("splitted_genome size before join=" + str(sys.getsizeof(splitted_genome) / 10 ** 9) + " GB")
    index = 0
    for i in range(0, len(localTables)):
        if localTables[i].shape[0] > 1:
            try:
                dataset = np.load(outputDir + '/dataset_2d_' + str(i) + '.npy')
                for j in range(dataset.shape[0]):
                    splitted_genome[index] = dataset[j]
                    index += 1
                os.remove(outputDir + '/dataset_2d_' + str(i) + '.npy')
            except FileNotFoundError:
                print('WARNING: I could not find: ' + outputDir + '/dataset_2d_' + str(i) + '.npy')
            #print("splitted_genome size after join of i="+str(i)+" is " + str(sys.getsizeof(splitted_genome) / 10 ** 9) + " GB")
    localTables = None
    current, peak = tracemalloc.get_traced_memory()  # to track memory used
    print(
        f"Method: join_dataset.\n Current memory usage is {current / 10 ** 9}GB; Peak was {peak / 10 ** 9}GB")  # to track memory used
    tracemalloc.stop()  # to track memory used
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Splitting of input sequences done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Second step: Predict initial and end position of LTR-RTs in each chunk
    print('INFO: Predicting which genome chunk contains LTR-RTs...')
    start = time.time()
    position_predictions = Inpactor2_Detect(splitted_genome)
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: LTR-RTs containing prediction done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Third step: Extract sequences predicted as LTR-RTs
    print('INFO: Extracting sequences predicted as LTR-RTs...')
    start = time.time()
    tracemalloc.start()  # to track memory used
    # Run in parallel the extraction
    n = position_predictions.shape[0]
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    """splitted_genome_list = []
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
                                           outputDir, max_len_threshold, min_len_threshold, list_ids, tg_ca, TSD]) for x
                    in range(threads)]
    localTables = [p.get() for p in localresults]

    # to join local results of extracted sequences
    ltr_rts_predicted = np.zeros((1, 1))
    for i in range(0, len(localTables)):
        if localTables[i].shape[1] > 1:
            if ltr_rts_predicted.shape[0] == 1:
                try:
                    ltr_rts_predicted = np.load(outputDir + '/extracted_seqs_2d_' + str(i) + '.npy')
                    os.remove(outputDir + '/extracted_seqs_2d_' + str(i) + '.npy')
                except FileNotFoundError:
                    print('WARNING: I could not find: ' + outputDir + '/extracted_seqs_2d_' + str(i) + '.npy')
            else:
                try:
                    ltr_rts_predicted = np.concatenate(
                    (ltr_rts_predicted, np.load(outputDir + '/extracted_seqs_2d_' + str(i) + '.npy')), axis=0)
                    os.remove(outputDir + '/extracted_seqs_2d_' + str(i) + '.npy')
                except FileNotFoundError:
                    print('WARNING: I could not find: ' + outputDir + '/extracted_seqs_2d_' + str(i) + '.npy')"""
    ltr_rts_predicted = sequences_extractor(splitted_genome, position_predictions, 1, seqs_per_procs, n, remain,
                                           outputDir, max_len_threshold, min_len_threshold, list_ids, tg_ca, TSD)
    print(ltr_rts_predicted.shape)
    # to join local results of predicted IDs
    ids_predicted = []
    for i in range(1):
        try:
            IDsfile = open(outputDir + '/predicted_ids_' + str(i) + '.txt', 'r')
            lines = IDsfile.readlines()
            for line in lines:
                ids_predicted.append(line.replace('\n', ''))
            IDsfile.close()
            os.remove(outputDir + '/predicted_ids_' + str(i) + '.txt')
        except FileNotFoundError:
            print('WARNING: I could not find: ' + outputDir + '/predicted_ids_' + str(i) + '.txt')

    splitted_genome_list = None  # to clean unusable variable
    position_predictions = None  # to clean unusable variable
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Extraction done!!!! [time=' + str(finish - start) + ']')
    current, peak = tracemalloc.get_traced_memory()  # to track memory used
    print(
        f"Method: create_dataset.\n Current memory usage is {current / 10 ** 9}GB; Peak was {peak / 10 ** 9}GB")  # to track memory used
    # print("Local tables size="+str(sys.getsizeof(localTables) / 10 ** 9)+" GB")
    tracemalloc.stop()  # to track memory used

    if ltr_rts_predicted.shape[0] == 0:
        print('WARNING: There is no LTR retrotransposons that satisfy the conditions after structural filtration, '
              'try modifying the parameters -m, -n, -i, and -d ....')
        sys.exit(0)

    ##################################################################################
    # Fourth step: k-mer Counting (1<=k<=6) from sequences using a DNN
    print('INFO: Counting k-mer frequencies using a DNN...')
    start = time.time()
    kmer_counts = Inpactor2_kmer(ltr_rts_predicted, batch_size)
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: K-mer counting done!!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Fifth step: Filter sequences that are not full-length with a FNN.
    if curation:
        print('INFO: Filtering non-intact LTR-retrotransposons...')
        start = time.time()
        filtered_seqs, ids_predicted = Inpactor2_Filter(kmer_counts, ids_predicted)
        kmer_counts = None
        finish = time.time()
        total_time.append(finish - start)
        print('INFO: Filtering done!!!! [time=' + str(finish - start) + ']')

        if filtered_seqs.shape[0] == 0:
            print('WARNING: There is no LTR retrotransposons that satisfy the conditions after curation, try to re-run '
                  'Inpactor2 with the option -c no ....')
            sys.exit(0)
    else:
        filtered_seqs = kmer_counts
        kmer_counts = None

    ##################################################################################
    # Sixth step: Predict the lineage from the pre-trained DNN.
    print('INFO: Predicting the lineages from sequences...')
    start = time.time()
    predictions, percentajes = Inpactor2_Class(filtered_seqs)
    filtered_seqs = None
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Prediction of the lineages from sequences done!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Seventh step: Creating the description file of the predictions made by the DNNs
    print('INFO: Creating the prediction descriptions file...')
    start = time.time()
    finalIds = create_bed_file(ids_predicted, predictions, percentajes, outputDir, curation)
    finish = time.time()
    total_time.append(finish - start)
    print('INFO: Creating output file done!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Eighth step: Creating fasta file with LTR retrotransposons classified
    print('INFO: Creating LTR-retrotransposon library...')
    start = time.time()

    # Run in parallel the fasta creation
    n = len(finalIds)
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads

    splitted_sequences_list = []
    for i in range(threads):
        if i < remain:
            init = i * (seqs_per_procs + 1)
            end = init + seqs_per_procs + 1
        else:
            init = i * seqs_per_procs + remain
            end = n if init + seqs_per_procs > n else init + seqs_per_procs
        splitted_sequences_list.append(ltr_rts_predicted[init:end, :, :])

    pool = multiprocessing.Pool(processes=threads)
    localresults = [pool.apply_async(create_fasta_file,
                                     args=[splitted_sequences_list[x], finalIds, x, seqs_per_procs, n, remain,
                                           outputDir, curation]) for x in
                    range(threads)]
    splitted_sequences_list = None  # clean unusable variable
    localSequences = [p.get() for p in localresults]
    outputFile = open(outputDir + '/Inpactor2_library.fasta', 'w')
    for i in range(threads):
        filei = open(outputDir + '/Inpactor2_library_' + str(i) + '.fasta', 'r')
        lines = filei.readlines()
        for line in lines:
            outputFile.write(line)
        filei.close()
        try:
            os.remove(outputDir + '/Inpactor2_library_' + str(i) + '.fasta')
        except:
            print('I cannot delete the file: ' + outputDir + '/Inpactor2_library_' + str(i) + '.fasta')
    outputFile.close()
    finish = time.time()
    total_time.append(finish - start)

    ltr_rts_predicted = None  # clean unusable variable
    localSequences = None  # clean unusable variable
    finalIds = None  # clean unusable variable
    predictions = None  # clean unusable variable
    ids_predicted = None  # clean unusable variable
    percentajes = None  # clean unusable variable

    print('INFO: Library created!!! [time=' + str(finish - start) + ']')

    ##################################################################################
    # Ninth step: Annotating LTR-RTs using RepeatMasker
    if annotate.upper() == 'YES':
        print('INFO: Annotating LTR-retrotranposons with RepeatMasker...')
        start = time.time()
        result_command = os.system(
            'RepeatMasker -pa ' + str(threads) + ' -lib ' + outputDir + '/Inpactor2_library.fasta '
                                                                        '-dir ' + outputDir + ' -gff -nolow -no_is -norna ' + file)
        if result_command != 0:
            print('FATAL ERROR: RepeatMasker failed!!!')
        else:
            finish = time.time()
            total_time.append(finish - start)
            print('INFO: Annotation done!!! [time=' + str(finish - start) + ']')

    print('INFO: Inpactor2 execution done successfully [total time='+str(sum(total_time))+']')
