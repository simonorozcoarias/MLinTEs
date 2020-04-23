from Bio import SeqIO
import sys
from random import randint

def extractSeqs(diccInput, seqfile):
	lineageDicc={}
	for seq in SeqIO.parse(seqfile, 'fasta'):
		lin = str(seq.id).split("-")[0]
		if lin in lineageDicc.keys():
			lineageDicc[lin].append(str(seq.id))
		else:
			lineageDicc[lin] = [str(seq.id)]

	selectedFile = open(seqfile+".selected", "w")
	for key in diccInput.keys():
		print("doing "+key)
		marks = []
		if diccInput[key] <= len(lineageDicc[key]):
			while len(marks) < diccInput[key]:
				print("generating "+str(len(marks)) + " of "+str(diccInput[key]))
				r = randint(0, len(lineageDicc[key]) - 1)
				if r not in marks:
					marks.append(r)
					sequence = [str(x.seq) for x in SeqIO.parse(seqfile, 'fasta') if str(x.id) == lineageDicc[key][r]]
					selectedFile.write(">"+lineageDicc[key][r]+"\n"+sequence[0]+"\n")
		else:
			for seqid in lineageDicc[key]:
				sequence = [str(x.seq) for x in SeqIO.parse(seqfile, 'fasta') if str(x.id) == seqid]
				selectedFile.write(">"+seqid+"\n"+sequence[0]+"\n")


if __name__ == '__main__':
	diccInput = {"ALE": 53, "ANGELA": 32, "ATHILA": 107, "BIANCA": 36, "CRM": 101, "DEL": 162, "GALADRIEL": 27, "IVANA": 7,
				"ORYCO": 438, "REINA": 551, "RETROFIT": 781, "SIRE": 63, "TAT": 203, "TORK": 281}
	seqfile = sys.argv[1]
	extractSeqs(diccInput, seqfile)
