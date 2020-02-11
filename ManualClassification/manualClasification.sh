#!/bin/bash

# This script identifies TE domains from RexDB, LTR length and LTR similarity and writes them in a semicolon-tabular file

DB=cores-database-wickercode.Lineage_Bianca-REXDB.fa
seqfile=$1
threads=$2
rm -f ${seqfile}.tab
for i in `grep ">" ${seqfile} | sed 's/>//g'`
do
  seqret -sequence ${seqfile}:${i} -outseq ${i}.RT.fa
  #GAG
  BLASTRESULT_GAG=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'GAG_' | head -n 1 | cut -f 2`
	BLASTEVALUE_GAG=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'GAG_' | head -n 1 | cut -f 11`
	#RT
	BLASTRESULT_RT=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'RT_' | head -n 1 | cut -f 2`
	BLASTEVALUE_RT=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'RT_' | head -n 1 | cut -f 11`
	#INT
	BLASTRESULT_INT=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'INT_' | head -n 1 | cut -f 2`
	BLASTEVALUE_INT=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'INT_' | head -n 1 | cut -f 11`
	#RNaseH
	BLASTRESULT_RNASE=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'RNaseH_' | head -n 1 | cut -f 2`
	BLASTEVALUE_RNASE=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'RNaseH_' | head -n 1 | cut -f 11`
	#AP
	BLASTRESULT_AP=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'AP_' | head -n 1 | cut -f 2`
	BLASTEVALUE_AP=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'AP_' | head -n 1 | cut -f 11`
	#ENV
	BLASTRESULT_ENV=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'ENV_' | head -n 1 | cut -f 2`
  BLASTEVALUE_ENV=`blastall -p blastx -a $threads -d $DB -i ${i}.RT.fa -e 1e-4 -m8 | grep 'ENV_' | head -n 1 | cut -f 11`
  id=`echo $i`
  seqfile2=${i}.RT.fa
  len=`infoseq -sequence $seqfile2 -only -length -noheading | sed 's/ //g'`
  half=`echo $len / 2 | bc`
  extractseq -sequence $seqfile2 -region 1-$half -outseq ${seqfile2}.5prime.fa
  extractseq -sequence $seqfile2 -region $half-$len -outseq ${seqfile2}.3prime.fa
  makeblastdb -in ${seqfile2}.5prime.fa -dbtype nucl
  blastn -query ${seqfile2}.3prime.fa -db ${seqfile2}.5prime.fa -outfmt "6 pident length qstart qend sstart send" -evalue 1e-20 -out ${seqfile2}.blast -num_threads $threads
  percIden=`cut -f1 ${seqfile2}.blast | head -n 1`
  lenLTR=`cut -f2 ${seqfile2}.blast | head -n 1`
  echo "${id};GAG ${BLASTRESULT_GAG:-NO};E${BLASTEVALUE_GAG:-NO};RT ${BLASTRESULT_RT:-NO};E${BLASTEVALUE_RT:-NO};INT ${BLASTRESULT_INT:-NO};E${BLASTEVALUE_INT:-NO};RNaseH ${BLASTRESULT_RNASE:-NO};E${BLASTEVALUE_RNASE:-NO};AP ${BLASTRESULT_AP:-NO};E${BLASTEVALUE_AP:-NO};ENV ${BLASTRESULT_ENV:-NO};E${BLASTEVALUE_ENV:-NO};$len;$percIden;$lenLTR" >> ${seqfile}.tab
  rm -f ${seqfile2}.5prime.fa  ${seqfile2}.3prime.fa  ${seqfile2}.blast $seqfile2 ${seqfile2}.5prime.fa.n*
done

# awk -F ';' '{
#   split("", domains);
#   split($2,GAG," ");
#   if(GAG[2] != "NO"){
#     split(GAG[2], GAG_subf, "#");
#     split(GAG_subf[2],GAG_name,"+");
#     domains[toupper(GAG_name[2])] += 1; 
#     print "GAG Found: "GAG_name[2];
#   }
#   split($4,RTT," ");
#   if(RTT[2] != "NO"){
#     split(RTT[2], RT_subf, "#");
#     split(RT_subf[2],RT_name,"+");
#     domains[toupper(RT_name[2])] += 1;
#     print "RT Found: "RT_name[2];
#   }
#   split($6,INT," ");
#   if(INT[2] != "NO"){
#     split(INT[2], INT_subf, "#");
#     split(INT_subf[2],INT_name,"+");
#     domains[toupper(INT_name[2])] += 1 
#     print "INT Found: "INT_name[2];
#   }
#   split($8,RNAS," ");
#   if(RNAS[2] != "NO"){
#     split(RNAS[2], RNAS_subf, "#");
#     split(RNAS_subf[2],RNAS_name,"+");
#     domains[toupper(RNAS_name[2])] += 1 
#     print "RNAS Found: "RNAS_name[2];
#   }
#   split($10,AP," ");
#   if(AP[2] != "NO"){
#     split(AP[2], AP_subf, "#");
#     split(AP_subf[2],AP_name,"+");
#     domains[toupper(AP_name[2])] += 1 
#     print "AP Found: "AP_name[2];
#   }
#   split($12,ENV," ");
#   if(ENV[2] != "NO"){
#     split(ENV[2], ENV_subf, "#");
#     split(ENV_subf[2],ENV_name,"+");
#     domains[toupper(ENV_name[2])] += 1 
#     print "ENV Found: "ENV_name[2];
#   }
#   len = 0;
#   max = 0;
#   no_fam = 0;
#   print "Element: "$1" has the following domines";
#   for (dom in domains){
#     if(domains[dom]>=max && domains[dom] > 0){
#       if(domains[dom] == max){
#         no_fam = 1;
#       }
#       max = domains[dom];
#       family = dom;
#     }
#     print "domine: "dom" num: "domains[dom];
#     len += 1;
#   }
#   if(len == 0 || no_fam == 1 || len < 3){
#     family = "NO_FAMILY"
#   }
#   print "family: "family;
#   print $0
#   print $0 >> family".tab.family";
# }' ${seqfile}.tab
