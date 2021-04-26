#1-gram
import sys
training_file=open("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/wiki-en-train.word","r")
counts=dict()
total_count=0
for line in training_file:
    line=line.strip()
    if len(line)!=0:
        sentence=line.split(" ")
        sentence.append("</s>")
        for word in sentence:
            if word in counts:
                counts[word]+=1
                total_count+=1
            else:
                counts[word]=1
                total_count+=1
training_file.close()

ans=open("model_file.word","w")
for word in counts:
    p=counts[word]/total_count
    ans.write(str(word)+" "+str(p)+"\n")
ans.close()

