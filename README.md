# A computational Architecture to identify and classify LTR retrotransposons in plant genomes.
Ph.D. Thesis of Simon Orozco-Arias
Advisor: Gustavo Isaza, Ph.D.
Co-Advisor: Romain Guyot, Ph.D.

## The organization of this repository is the following:

* [CodingSchemes](#CodingSchemes)  
* [DL_Exps](#DL_Exps)  
* [Inpactor2](#Inpactor2)  
* [ML_Exps](#ML_Exps) 
* [ManualClassification](#ManualClassification) 
* [Original_DBs](#Original_DBs) 
* [Position_detection](#Position_detection) 
* [Undergrate_projects](#Undergrate_projects) 
* [k_mer_gen_exps](#k_mer_gen_exps) 
* [retraining_coffee](#retraining_coffee)
* [Published papers](#papers) 
* [Other useful resources](#resources) 

## CodingSchemes folder
<a name="introduction"/>
Scripts used to convert DNA sequences into numerical respresentatons.

## DL_Exps folder
<a name="DL_Exps"/>
Neural network experiments done in the thesis to identify, filtering and classify LTR-retrotransposons

## Inpactor2 folder
<a name="Inpactor2"/>
Folder containing all versions of the software Inpactor2. The last (stable) version is available at: https://github.com/simonorozcoarias/Inpactor2

## ML_Exps folder
<a name="ML_Exps"/>
Scripts used to test different coding schemes, pre-processing techniques and ML algorithms.

## ManualClassification folder
<a name="ManualClassification"/>
Scripts to classify LTR retrotransposons using homology-based method based on the enzymatic domains present in those elements.

## Original_DBs folder
<a name="Original_DBs"/>
Some databases used in ML experiments

## Position_detection folder
<a name="Position_detection"/>
Scripts and neural network architectures used in the detection problem.

## Undergrate_projects folder
<a name="Undergrate_projects"/>
Folder containing undergraduate projects related with this PhD thesis.

## k_mer_gen_exps folder
<a name="k_mer_gen_exps"/>
Scripts and neural networks used to find a solution to the bottleneck of calculating k-mers.

## retraining_coffee folder
<a name="retraining_coffee"/>
This folder contains the jupyter notebook used to train Inpactor2 to specialize it to the coffea genus

## Publish papers
<a name="papers"/>

* Orozco-Arias, S., Liu, J., Tabares-Soto, R., Ceballos, D., Silva Domingues, D., Garavito, A., ... & Guyot, R. (**2018**). Inpactor, integrated and parallel analyzer and classifier of LTR retrotransposons and its application for pineapple LTR retrotransposons diversity and dynamics. Biology, 7(2), 32.
* Orozco-Arias, S., Isaza, G., & Guyot, R. (**2019**). Retrotransposons in plant genomes: structure, identification, and classification through bioinformatics and machine learning. International journal of molecular sciences, 20(15), 3837.
* Orozco-Arias, S., Isaza, G., Guyot, R., & Tabares-Soto, R. (**2019**). A systematic review of the application of machine learning in the detection and classification of transposable elements. PeerJ, 7, e8311.
* Orozco-Arias, S., Piña, J. S., Tabares-Soto, R., Castillo-Ossa, L. F., Guyot, R., & Isaza, G. (**2020**). Measuring performance metrics of machine learning algorithms for detecting and classifying transposable elements. Processes, 8(6), 638.
* Orozco-Arias, S., Jaimes, P. A., Candamil, M. S., Jiménez-Varón, C. F., Tabares-Soto, R., Isaza, G., & Guyot, R. (**2021**). InpactorDB: a classified lineage-level plant LTR retrotransposon reference library for free-alignment methods based on machine learning. Genes, 12(2), 190.
* Orozco-Arias, S., Candamil-Cortés, M. S., Jaimes, P. A., Piña, J. S., Tabares-Soto, R., Guyot, R., & Isaza, G. (**2021**). K-mer-based machine learning method to classify LTR-retrotransposons in plant genomes. PeerJ, 9, e11456.
* Orozco-Arias, S., Candamil-Cortes, M. S., Jaimes, P. A., Valencia-Castrillon, E., Tabares-Soto, R., Guyot, R., & Isaza, G. (**2021**). Deep Neural Network to Curate LTR Retrotransposon Libraries from Plant Genomes. In International Conference on Practical Applications of Computational Biology & Bioinformatics (pp. 85-94). Springer, Cham.
* Orozco-Arias, S., Candamil-Cortés, M. S., Valencia-Castrillón, E., Jaimes, P. A., Tobón Orozco, N., Arias-Mendoza, M., Tabares-Soto, R., Guyot, R., & Isaza, G. (**2021**). SENMAP: A Convolutional Neural Network Architecture for Curation of LTR-RT Libraries from Plant Genomes. In 2021 IEEE 2nd International Congress of Biomedical Engineering and Bioengineering (CI-IB&BI) (pp. 1-4). IEEE.

## Other useful resourcers
<a name="resources"/>

* Inpactor version 1 (non-DL implementation): [Inpactor V1 github](https://github.com/simonorozcoarias/Inpactor)
* LTR retrotransposon classification experiments using ML: [ML experiments github](https://github.com/simonorozcoarias/MachineLearningInTEs)
* Plant LTR retrotransposon reference library: [InpactorDB dataset](https://zenodo.org/record/5816833#.YdRXUXWZNH4)
* Dataset of genomic features other than LTR-RTs: [Negative Instances dataset](https://zenodo.org/record/4543905#.YdRXpnWZNH4)
