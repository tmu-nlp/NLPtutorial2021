import sys
def make_corpus(my_file):
    corpus=dict()
    for line in my_file:
        line=line.strip()
        if len(line)!=0:
            s=line.split(" ")
            for w in s:
                if w in corpus:
                    corpus[w]+=1
                else:
                    corpus[w]=1
    return corpus

def word_count(corpus):
    ans=open("00-answer.txt","w")
    ans.write(str(list(corpus.items())))


my_file=open(sys.argv[1],"r")
word_count(make_corpus(my_file))
my_file.close()






