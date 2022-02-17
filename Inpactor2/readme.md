# the new version of Inpactor, an annotator that uses DNN for detection and classification of LTR retrotransposons in plant genomes

## Current Version stable: https://github.com/simonorozcoarias/Inpactor2
Version history: 

version 2.0 (Bugs corrected and deletion of unuse code)

version 1.9 (Improving the modularity using master and slave functions separately)

version 1.8 (Added the verbose option)

version 1.7 (Added the cycles analysis option)

version 1.6 (Added the Non-maximal suppression step and replace BLASTn by LTR_Finder for detection)

version 1.5 (Creation of functions for each step, deleting those lines from main function)

version 1.4 (Added the pre-filter step checking if the seqs contain non-nucleotic characters)

version 1.3 (Integration of RepeatMasker to annotate the newly created library) 

version 1.2 (Detection using Inpactor2_Detect and using BLASTn for the starting and ending positions, counting k-mer using a CNN, and filtering with FNN)

version 0.7 (Utilization of EDTA to detect LTR-RTs)

version 0.6 (MPI implementation using MPI4py python package)

version 0.5 (k-mer counting improved using dynamic programming)

version 0.4 (Solved issue with huge temporal file in the splitting step)

version 0.3 (mode selection [slow or fast], output in bed and fasta formats, and posibility to use a pre-calculated k-mer file)

version 0.2 (Genomic splitting, k-mer generation (inefficient), lineages prediction, and output in bed format)

version 0.1 (two step without genomic splitting)
