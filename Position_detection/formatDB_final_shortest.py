#!/bin/env pyhton

import sys
import os
import Bio
from Bio import SeqIO
import itertools
import time
import multiprocessing

"""
This function calculates the minimum length found in the dataset
"""
def minLength(fastafile):
    minLen = 50000
    numSeqs = 0
    idseq = ""
    for te in SeqIO.parse(fastafile, "fasta"):
        if len(te.seq) < minLen:
            minLen = len(te.seq)
            idseq = te.id
        numSeqs += 1
    print("shortest seq: "+idseq)
    return minLen, numSeqs


"""
This function calculates k-mer frequencies in parallel mode.
"""
def kmerDB(file, id, seqs_per_procs, kmers, TEids, TEseqs, n, remain, minLen):
    if id < remain:
        init = id * (seqs_per_procs + 1)
        end = init + seqs_per_procs + 1
    else:
        init = id * seqs_per_procs + remain
        end = init + seqs_per_procs
    print("running in process " + str(id) + " init=" + str(init) + " end=" + str(end))
    resultFile = open(file + '.' + multiprocessing.current_process().name, 'w')

    while init < end and init < n:
        order = -1
        """if str(TEids[init]).upper().find("RLC_") != -1 or str(TEids[init]).upper().find("COPIA") != -1:
            order = 1
        elif str(TEids[init]).upper().find("RLG_") != -1 or str(TEids[init]).upper().find("GYPSY") != -1:
            order = 2"""

        # Lineages from Copia
        """if str(TEids[init]).upper().find("ALE-") != -1 or str(TEids[init]).upper().find("RETROFIT-") != -1 or str(TEids[init]).upper().find("ALESIA-") != -1 or str(TEids[init]).upper().find("ANGELA-") != -1 		or str(TEids[init]).upper().find("BIANCA-") != -1 or str(TEids[init]).upper().find("BRYCO-") != -1 or str(TEids[init]).upper().find("LYCO-") != -1 or str(TEids[init]).upper().find("GYMCO-") != -1 		or str(TEids[init]).upper().find("IKEROS-") != -1 or str(TEids[init]).upper().find("IVANA-") != -1 or str(TEids[init]).upper().find("ORYCO-") != -1 or str(TEids[init]).upper().find("OSSER-") != -1 or str(TEids[init]).upper().find("TAR-") != -1 or str(TEids[init]).upper().find("TORK-") != -1 or str(TEids[init]).upper().find("SIRE-") != -1:
            order = 1
        # Lineages from Gypsy
        elif str(TEids[init]).upper().find("CRM-") != -1 or str(TEids[init]).upper().find("CHLAMYVIR-") != -1 or str(TEids[init]).upper().find("GALADRIEL-") != -1 or str(TEids[init]).upper().find("REINA-") != -1 		or str(TEids[init]).upper().find("TEKAY-") != -1 or str(TEids[init]).upper().find("DEL-") != -1 or str(TEids[init]).upper().find("ATHILA-") != -1 or str(TEids[init]).upper().find("OGRE-") != -1 		or str(TEids[init]).upper().find("RETAND-") != -1 or str(TEids[init]).upper().find("PHYGY-") != -1 or str(TEids[init]).upper().find("SELGY-") != -1:
            order = 0"""
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
            if len(TEseqs[init]) <= minLen:
                frequencies = [0 for x in range(len(kmers))]
                for i in range(len(TEseqs[init])):
                    for l in range(1, 7):
                        if i + l < len(TEseqs[init]):
                            if TEseqs[init][i:i + l].upper().find("N") == -1 and TEseqs[init][i:i + l].upper() in kmers:
                                index = kmers.index(TEseqs[init][i:i + l].upper())
                                frequencies[index] += 1
                # print (TEids[init])
                frequenciesStrings = [str(x) for x in frequencies]
                resultFile.write(str(order) + ',' + ','.join(frequenciesStrings) + '\n')
                # resultMap[TEids[init]] = str(order)+','+','.join(frequenciesStrings)
                init += 1
            else:
                for pos in range(0, len(TEseqs[init]), minLen):
                    if pos + minLen > len(TEseqs[init]):
                        actualEnd = len(TEseqs[init])
                        actualStart = len(TEseqs[init]) - minLen
                    else:
                        actualEnd = pos + minLen
                        actualStart = pos
                    newSeq = str(TEseqs[init])[actualStart: actualEnd]
                    frequencies = [0 for x in range(len(kmers))]
                    for i in range(len(newSeq)):
                        for l in range(1, 7):
                            if i + l < len(newSeq):
                                if newSeq[i:i + l].upper().find("N") == -1 and newSeq[i:i + l].upper() in kmers:
                                    index = kmers.index(newSeq[i:i + l].upper())
                                    frequencies[index] += 1
                    # print (TEids[init])
                    frequenciesStrings = [str(x) for x in frequencies]
                    resultFile.write(str(order) + ',' + ','.join(frequenciesStrings) + '\n')
                    # resultMap[TEids[init]] = str(order)+','+','.join(frequenciesStrings)
                init += 1
    resultFile.close()
    print("Process done in " + str(id))


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
    fasta_file = sys.argv[1]
    #filter(fasta_file)
    minLen, numSeqs = minLength(fasta_file+".filtered")
    # kmer features calculator in parallel mode
    # number of threads to calculate k-mer frequencies in parallel.
    threads = 64
    start_time = time.time()
    """kmers = ['A', 'T', 'AAAAAA', 'ATAT', 'AGGGGG', 'CCCCCT', 'TTTTTT', 'AGCT', 'GATC', 'GATGA', 'ACATA', 'TATGT',
             'CTTC', 'CCTT', 'CTTG', 'G', 'ATGT', 'ACAT', 'TCT', 'TTAAT', 'GA', 'AAAG', 'CTAC', 'CCAT', 'TATA', 'CTAG',
             'TAAC', 'GGGGAG', 'CATC', 'GAA', 'CTT', 'GTTA', 'TGCC', 'TCTC', 'CTTTTT', 'AATC', 'CCAAT', 'TATAT', 'CCA',
             'GTAC', 'TCATC', 'AGA', 'CT', 'GGGGGA', 'GTA', 'ATATA', 'C', 'ATGC', 'CTTT', 'AAAAA', 'TGGGG', 'AAGG',
             'GGCC', 'AAG', 'GATG', 'AAAGG', 'GAAG', 'TC', 'GC', 'TGTA', 'TTGA', 'CTG', 'ATC', 'TAC', 'GGC', 'CATG',
             'TTC', 'CAAG', 'AAAAAG', 'CAG', 'TAGG', 'AATGC', 'CTTGT', 'CCGG', 'TATG', 'GAATT', 'CCAA', 'CCTTT',
             'TCCCCC', 'TACAT', 'ATATC', 'AT', 'GAGA', 'TTAA', 'GTAG', 'TACC', 'TCAA', 'TCTG', 'TGG', 'AGTT', 'GTAA',
             'TGC', 'GGG', 'TTTTT', 'TGGGGG', 'GATAT', 'TAAG', 'AAGC', 'CATTA', 'CTGG', 'GAAGA', 'ACGT', 'AAGTC',
             'AACTC', 'GGTGG', 'CATGT', 'CCCC', 'GCAT', 'CCAATC', 'TCTTC', 'GTGT', 'TTTT', 'CCC', 'CAAGT', 'TGCA',
             'ATGTA', 'GGAAA', 'TTAC', 'TTGGC', 'TTTGA', 'TACA', 'AATTC', 'GATTGG', 'ATGG', 'AACT', 'GAT', 'TCGA',
             'GAG', 'TAG', 'ACTTG', 'AG', 'CTCCCC', 'CAACT', 'TTCTT', 'TCTTG', 'CAGT', 'TGGA', 'TAGC', 'ATTGG', 'ATCT',
             'GAGTT', 'TGTTT', 'GCTT', 'GCC', 'GAGC', 'GGTTTT', 'ATTAAT', 'GTTATG', 'TCATCA', 'GATT', 'CTTTT', 'AAAACC',
             'AAGCA', 'GAAAG', 'ACCC', 'GAAGT', 'CATAA', 'GATATC', 'TGGTAT', 'GGTA', 'TTCT', 'ATGCC', 'GGGT', 'AGTGGG',
             'ATA', 'GCATT', 'CCATG', 'TTGGG', 'CACA', 'ATG', 'CCCCCA', 'TTATG', 'GCCAA', 'CAAAG', 'CTCT', 'TGATGA',
             'CCACC', 'GAATC', 'TCTT', 'ATATCT', 'CCAC', 'CATAAC', 'GGTATC', 'GCT', 'TTGC', 'ACAG', 'GACTT', 'AGAT',
             'TACATG', 'TCG', 'AAGAA', 'ATGTAT', 'TTGAG', 'GACT', 'CCCACT', 'AAAA', 'GGAA', 'CTAAG', 'AAACA', 'ATGTC',
             'TGGCC', 'CATCT', 'AGTC', 'TCACA', 'ATGAT', 'ACTG', 'CC', 'CATAT', 'AAAGA', 'CCTTA', 'TATGG', 'CCAG',
             'CTTGA', 'CATGG', 'GGGTA', 'AATCA', 'CATA', 'GCTTC', 'ATTAA', 'CCCCA', 'GGCA', 'TTAT', 'ACT', 'TCCC',
             'GAAATG', 'ATTC', 'GGCCA', 'TGAAG', 'CTATG', 'AAGGG', 'AC', 'AGC', 'CCATA', 'AGGG', 'GCTTG', 'AAGAAG',
             'ATCA', 'TA', 'GAAT', 'CACC', 'TCAG', 'GGGG', 'ATGATG', 'AAGT', 'AGAC', 'CAGA', 'AGT', 'CCGA', 'CCTA',
             'CTGA', 'ATATG', 'GGAT', 'CAAT', 'CTTTG', 'CAAAAA', 'TCGG', 'ACAC', 'GATATG', 'AGTA', 'ATCGA', 'TGGAA',
             'CTCTG', 'TACT', 'ATTAT', 'TTGGCC', 'GG', 'AGTGG', 'TGCAT', 'AATG', 'CCAAG', 'CTC', 'TGCTT', 'AGAG',
             'GAAAA', 'GATGG', 'TTTCTT', 'CCCCC', 'TGTGT', 'AAC', 'CCTG', 'TCCT', 'TCAAA', 'CCCT', 'ATAC', 'TCA',
             'AAAAG', 'GCG', 'GGGGG', 'TGGCCA', 'CAAC', 'AAAAGA', 'GCATC', 'CGA', 'GAAC', 'AAGGC', 'TTTTTG', 'TTGTA',
             'AGAA', 'GATGAA', 'TGGAT', 'GGCGCC', 'ACAAG', 'GGTG', 'GATGC', 'TTTTGG', 'CTTAA', 'ACATG', 'ACCATG',
             'GGTAT', 'ACTC', 'TCAAG', 'TAGGA', 'CTCCAT', 'TAAT', 'CAA', 'AACTTT', 'TTACC', 'ATTG', 'GGAGT', 'ATGTTT',
             'GATTA', 'TTGCT', 'AGATAT', 'GAGG', 'CCACT', 'TGT', 'CTCC', 'CCT', 'CAC', 'GATCT', 'CCCAA', 'TGATG',
             'TGAGA', 'TGTAC', 'TGTG', 'GGAAG', 'GTTC', 'GGCT', 'CTCAA', 'CCAGAA', 'CTTCTT', 'CCCTT', 'GTCT', 'ACAAA',
             'AAAC', 'TGAGT', 'GGT', 'AGCTT', 'GCAC', 'TGGG', 'CCTAG', 'GAAAAA', 'AGCAA', 'ATAAT', 'CTTTC', 'AGATC',
             'GAGGG', 'GATTC', 'CAGC', 'ATATAT', 'GGA', 'TTG', 'GGTT', 'ACG', 'GCTA', 'TGAT', 'CTTCT', 'AGCC', 'TCCCC',
             'GTG', 'GTC', 'TGTAA', 'AGGA', 'GTGTG', 'TGGAG', 'GTGGGA', 'ACATGT', 'GT', 'TTCAT', 'TTGTT', 'CATATG',
             'TGGC', 'GGTAA', 'AGTTG', 'CCATGG', 'GAC', 'TGGTA', 'AGAGA', 'TTCTGG', 'TTGG', 'TAATG', 'TACCC', 'GGGC',
             'AATGG', 'CACG', 'GTT', 'TTTC', 'CTTCA', 'TT', 'GTGGGG', 'ACTCC', 'CTA', 'GGAG', 'TGTGA', 'ATGGTA', 'ACTT',
             'GTGG', 'TGA', 'CATTG', 'CCCTCC', 'TTTAA', 'AGG', 'GAAA', 'GCAG', 'AGAGAG', 'TGCTCA', 'GATAAT', 'CTTCTG',
             'TTGCC', 'CAGAG', 'AGCA', 'CCAAAA', 'ACC', 'TGATGC', 'CTGT', 'GCA', 'CAAGC', 'GTATCA', 'TCTCA', 'CGCCG',
             'CATATC', 'GCCA', 'CATCC', 'CAAGCT', 'ATTTC', 'CACTG', 'AATTAA', 'ATGCAT', 'ATCAGA', 'TCGT', 'TCCAC',
             'ACCACA', 'CCTTC', 'GCTTGG', 'ATAA', 'GTTT', 'AGGC', 'AATCAT', 'ACTGA', 'CTTCC', 'CATGGT', 'GGAGG',
             'GCTTT', 'TGAAGA', 'AACTG', 'AACAT', 'GACA', 'TTTGAA', 'AAGA', 'TTCCA', 'TTCC', 'AAATCT', 'GGAC', 'ATGAA',
             'TGCTG', 'TCTA', 'CAGG', 'CATT', 'ACA', 'GCTTTT', 'GGAGTG', 'TTAG', 'TTACA', 'GTTAT', 'GCAAC', 'CCATT',
             'ACACA', 'CAAGCA', 'AAGCT', 'TCTCT', 'GTCAG', 'CTCCA', 'ACTAC', 'TTGGAG', 'CAGTT', 'CGTA', 'TCTTT',
             'CAAGA', 'CATAG', 'AAGAAA', 'GATGAT', 'GAAGC', 'TAGA', 'CTGAA', 'AAAAAC', 'ATACAT', 'GGTTC', 'TATATA',
             'GGTTT', 'CAT', 'CTCTC', 'GTGC', 'ACCG', 'CACT', 'CATCCA', 'TTTTG', 'TTGTC', 'GATCTA']"""
    kmers = []
    for k in range(1,7):
        for item in itertools.product('ACGT', repeat=k):
            kmers.append(''.join(item))
    TEids = []
    TEseqs = []
    n = 0
    for te in SeqIO.parse(fasta_file+".filtered", "fasta"):
        TEids.append(te.id)
        TEseqs.append(te.seq)
        n += 1
    seqs_per_procs = int(n / threads) + 1
    remain = n % threads
    processes = [multiprocessing.Process(target=kmerDB, args=[fasta_file+".filtered", x, seqs_per_procs, kmers, TEids, TEseqs, n, remain, 1000])
                 for x in range(threads)]
    [process.start() for process in processes]
    [process.join() for process in processes]

    finalFile = open(fasta_file+".filtered" + '.1kb_kmers', 'w')
    finalFile.write('Label,' + ','.join(kmers) + '\n')
    for i in range(1, threads + 1):
        filei = open(fasta_file+".filtered" + '.Process-' + str(i), 'r')
        lines = filei.readlines()
        for line in lines:
            finalFile.write(line)
        filei.close()
        os.remove(fasta_file+".filtered" + '.Process-' + str(i))
    finalFile.close()
    end_time = time.time()
    print("Threads time=", end_time - start_time)
