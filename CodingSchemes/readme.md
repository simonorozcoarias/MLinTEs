# CodingSchemes folder

## The organization of this folder is the following:

### 2dconversion.py
This script converts a DNA sequence into different 2D representations using as a basis the representation of a point (see figure below). This script contains the following 2D representations:
1. Self-replication: consists of replicating the DNA sequence until the desired length is reached (in this case the maximum length of the sequence in the dataset).
2. Zeros: Consists in filling the representation of the DNA sequence with zeros.
3. NNs: Consists in filling the representation of the DNA sequence with Ns.
4. Center: Consists of placing the DNA sequence in the center of the 2D representations and filling both sides with zeros.
5. Ones: Consists of filling the DNA sequence representation with ones.
6. TERL: Consists in to add another row to the 2D representation named as border, and fill it with ones at the end of the sequences
7. Positional k-mer: Consists in to calculate the frequencies of 508 k-mers in 508 sections of the sequence. The final result is a matrix of 508 rows (sequence' sections) and 508 columns (k-mer frequencies)

To execute the script:

```
python3 2dconversion.py fasta_file.fasta representation_id
```
Where *fasta_file.fasta* is your own file in fasta format with the LTR retrotransposons to convert and *representation_id* is the number of the representation that you desire (1 to 7 see list above).

### Positional_kmers_conversion_parallel.py
This script convert a DNA sequence to the Positional k-mer representation (see description in the 2dconversion.py script) in parallel.

To execute the script:

```
python3 2dconversion.py fasta_file.fasta number_threads
```
Where *fasta_file.fasta* is your own file in fasta format with the LTR retrotransposons to convert and *number_threads* is the number of threads available in your own computer.
