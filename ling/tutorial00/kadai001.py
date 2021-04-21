import sys
my_file=open(sys.argv[1],"r")
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

my_file.close()

ans=open("00-answer.txt","w")
ans.write(str(list(corpus.items())))




