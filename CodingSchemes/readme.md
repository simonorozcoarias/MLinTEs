# CodingSchemes folder

## The organization of this folder is the following:

### 2dconversion.py
This script converts a DNA sequence into different 2D representations using as a basis the representation of a point (see figure below). This script contains the following 2D representations:

<p align="center">
  <img src="https://github.com/simonorozcoarias/MLinTEs/blob/master/CodingSchemes/2d%20coding%20schemes.png">
</p>

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

### extractNegativeInstancesRandom.py
Script used to extract randomly sequences from a dataset. The number of sequences to extract is given in a dictionary in line 33.

To execute the script:

```
python3 extractNegativeInstancesRandom.py fasta_file.fasta 
```
Where *fasta_file.fasta* is your own file in fasta format with the LTR retrotransposons.

### formatDB_final.py 
This script convert DNA sequences into numerical representations. A difference of the first script of this repo, these representation are in one dimention. The results of this script is a file for each of the seven numerical representations. The available numerical representations are describe as following:
1. DAX: {‘C’:0, ‘T’:1, ‘A’:2, ‘G’:3}
2. EIIP: {‘C’:0.1340, ‘T’:0.1335, ‘A’:0.1260, ‘G’:0.0806}
3. Complementary: {‘C’:-1, ‘T’:-2, ‘A’:2, ‘G’:1}
4. Enthalpy: {‘CC’:0.11, ‘TT’:0.091, ‘AA’:0.091, ‘GG’:0.11, ‘CT’:0.078, ‘TA’:0.06, ‘AG’:0.078, ‘CA’:0.058, ‘TG’:0.058, ‘CG’: 0.119, ‘TC’:0.056, ‘AT’:0.086, ‘GA’:0.056, ‘AC’:0.065, ‘GT’:0.065, ‘GC’:0.1111}
5. Galois(4): {‘CC’:0.0, ‘CT’:1.0, ‘CA’:2.0, ‘CG’:3.0, ‘TC’:4.0, ‘TT’:5.0, ‘TA’:6.0, ‘TG’:7.0, ‘AC:8.0, ‘AT: 9.0, ‘AA’:1.0, ‘AG:11.0, ‘GC’:12.0, ‘GT’:13.0, ‘GA’:14.0, ‘GG’:15.0 }
6. Physico-chemical properties: This approach calculate three properties such as average hydrogen bonding energy per base pair (bp), stacking energy (per bp), and solvation energy (per bp).
7. k-mer frequencies using k values between one and six.

To execute the script:
```
python3 formatDB_final.py fasta_file.fasta number_threads
```
Where *fasta_file.fasta* is your own file in fasta format with the LTR retrotransposons to convert and *number_threads* is the number of threads available in your own computer.

### step1_one_hot_conversion.py 
This script split the input sequences into sectios of 50,000 bases and then, converts them into one-hot numerical representation.

To execute the script:
```
python3 step1_one_hot_conversion.py fasta_file.fasta 
```
Where *fasta_file.fasta* is your own file in fasta format with the LTR retrotransposons.
