import sys
training_file=open(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-train.word","r",encoding='utf-8')
counts=dict()
context_counts=dict()
context_counts[""]=0#先に初期化しないと、1gramの確率が1以上になる
for line in training_file:
    line=line.strip()
    if len(line)!=0:
        sentence=line.split(" ")#スペースで単語を分ける
        sentence.append("</s>")#文末記号
        sentence.insert(0,"<s>")#文頭記号
        for i in range(1,len(sentence)-1):
            word=sentence[i-1]+" "+sentence[i]
            if word in counts:
                counts[word]+=1
            else:
                counts[word]=1

            if sentence[i-1] in context_counts:
                context_counts[sentence[i-1]]+=1
            else:
                context_counts[sentence[i-1]]=1

            if sentence[i] in counts:
                counts[sentence[i]]+=1
                context_counts[""]+=1
            else:
                counts[sentence[i]]=1
                context_counts[""]+=1
         
training_file.close()
ans=open("model_file_bigram.word","w",encoding='utf-8')
for ngram in counts:
    tmp=ngram.split(" ")
    tmp.pop(-1)
    context=" ".join(tmp)
    #print(context)
    prob=float(counts[ngram]/context_counts[context])
    ans.write(ngram+" "+str(prob)+"\n")
ans.close()

