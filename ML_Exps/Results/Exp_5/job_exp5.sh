#!/bin/bash
#
#SBATCH --job-name=Exp_5_MLinTEs
#SBATCH --output=Exp5.out
#SBATCH -e Exp5.err
#SBATCH -D .
#SBATCH -n 20
#SBATCH --partition=highmem

unset PYTHONPATH
source ~/.bashrc;
conda activate ML

python3 MLclassifier_experiment5.py ~/Doctorado/databasesForML/finalDatabases/lineages_classification/PGSB/selfReplication/PGSB_0805219_cleaned.fasta_final 24 
