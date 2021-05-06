import sys
n=int(input("input n(n<=4): "))

training_file=open(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-train.word","r",encoding='utf-8')

def train(training_file,n):
    counts=dict()
    context_counts=dict()
    context_counts[""]=0
    for line in training_file:
        line=line.strip()
        if len(line)!=0:
            sentence=line.split(" ")#スペースで単語を分ける
            sentence.append("</s>")#文末記号
            sentence.insert(0,"<s>")#文頭記号
            for i in range(n-1,len(sentence)-1):
                tmp=n
                while tmp>=1:
                    word=" ".join(sentence[i-tmp+1:i+1])#target word bound with context words
                    context_word=" ".join(sentence[i-tmp+1:i])#only context words 

                    if word in counts:
                        counts[word]+=1
                    else:
                        counts[word]=1
                    if context_word in context_counts:
                        context_counts[context_word]+=1
                    else:
                        context_counts[context_word]=1
                    tmp-=1
               
    training_file.close()
    ans=open("model_file_ngram.word","w",encoding='utf-8')
    for ngram in counts:
        tmp=ngram.split(" ")
        tmp.pop(-1)
        context=" ".join(tmp)
        #print(context)
        prob=float(counts[ngram]/context_counts[context])
        ans.write(ngram+" "+str(prob)+"\n")
    ans.close()
    print("training finished")

train(training_file,n)

