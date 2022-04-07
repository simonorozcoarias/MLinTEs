# ManualClassification folder
This folder contains scripts in python to classify and filter LTR retrotransposons using a homology-based strategy.

## manualClasification.py
The main script. This would be run to performed the entire classification and filtering. This script need the following software:

* Biopython
* Censor (https://www.girinst.org/downloads/software/censor/)
* NCBI-BLAST
* EMBOSS

To execute the script:

```
python3 manualClasification.py fasta_file.fasta number_threads
```
Where *fasta_file.fasta* is your own file in fasta format with the LTR retrotransposons to classify and filtering and *number_threads* is the number of threads available in your own computer.

This script classifies first in superfamilies and then in lineages. Finally, it filters those sequences with the following conditions:

* predicted elements with domains from two different superfamilies (i.e. Gypsy and Copia)
* elements with domains belonging to two or more different lineages
* elements with lengths different than those reported by Gypsy Database (GyDB) with a tolerance of 20%
* incomplete elements which has less than three identified domains
* elements with insertions of TE class II (reported in Repbase and stored in the file repbase_20170127.fasta_class2). For this filter, Censor is required. 

*NOTE: This script was design to run over Linux systems.*

## manualClasification.sh
It is a auxiliary script written in bash language that performs a detection of different enzymatic domains and calculate some propierties need to classify and filtering the LTR retrotransposon sequences.
