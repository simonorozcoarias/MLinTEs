#!/bin/env python

import numpy as np
from Bio import SeqIO

def create_dataset(seqfile, total_win_len):
    print("loading files")
    pos_seqs = [x for x in SeqIO.parse(seqfile, 'fasta')]
    print("done")
    dataset = np.zeros((get_final_dataset_size(pos_seqs, total_win_len), 5, total_win_len), dtype=np.int16)
    j = 0
    for i in range(len(pos_seqs)):
        for k in range(0, len(str(pos_seqs[i].seq)), total_win_len):
            initial_pos = k
            end_pos = initial_pos + total_win_len
            final_seq = str(pos_seqs[i].seq)[initial_pos:end_pos]
            print(str(initial_pos)+" - "+str(end_pos))
            dataset[j] = one_hot(final_seq, total_win_len)
            j += 1

        if i % 100 == 0:
            print("doing "+str(i))

    print(dataset)
    print(dataset.shape)

    np.save("chr1.fna_50k.npy", dataset)


def get_final_dataset_size(seqfile, total_win_len):
    count = 0
    for i in range(len(seqfile)):
        for j in range(0, len(str(seqfile[i].seq)), total_win_len):
            count += 1
    return count

def one_hot(sequence, total_win_len):
    langu = ['A', 'C', 'G', 'T', 'N']
    posNucl = 0
    if len(sequence) < total_win_len:
        rest = ['N' for x in range(total_win_len - len(sequence))]
        sequence += ''.join(rest)

    rep2d = np.zeros((1, 5, len(sequence)), dtype=np.int16)

    for nucl in sequence:
        posLang = langu.index(nucl.upper())
        rep2d[0][posLang][posNucl] = 1
        posNucl += 1
    return rep2d

def filter(file):
    newFile = open(file+".filtered", "w")
    for te in SeqIO.parse(file, "fasta"):
        seq = str(te.seq)
        filterDna = [x for x in seq if x.upper() in ['A', 'C', 'G', 'T', 'N']]
        newSeq = "".join(filterDna)
        newFile.write(">"+str(te.id)+"\n"+newSeq+"\n")

if __name__ == '__main__':
    seqfile = sys.argv[1]
    total_win_len = 50000
    #filter(seqfile)
    create_dataset(seqfile, total_win_len)
