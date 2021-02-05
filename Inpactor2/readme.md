# the new version of Inpactor, an annotator that uses DNN for detection and classification of LTR retrotransposons in plant genomes

## Current version: 0.2 (Genomic splitting, k-mer generation (inefficient), lineages prediction, and output in bed format)
Version history:

version 0.1 (two step without genomic splitting)

To run:

download first the PCA model: https://drive.google.com/file/d/1UTO3YSPzs31tbDUP3K8r_QOQElFI8o3X/view?usp=sharing

Download the fasta file for test: https://drive.google.com/file/d/1RfreysQuHP_j1QqeiLP4urUDBzkjISj6/view?usp=sharing

Then, to run:

python3 Inpactor2.py -f InpactorDB_RepetDB_format.fa_1k
