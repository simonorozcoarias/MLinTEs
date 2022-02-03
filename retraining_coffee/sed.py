from Bio import SeqIO
import sys

def parser(fastafile, old, new):
	for seq in SeqIO.parse(fastafile, "fasta"):
		if old in str(seq.id):
			print(">"+seq.id+"#"+new+"\n"+seq.seq) 


if __name__ == '__main__':
	fastafile = sys.argv[1]
	old = sys.argv[2] 
	new = sys.argv[3]

	parser(fastafile, old, new)
