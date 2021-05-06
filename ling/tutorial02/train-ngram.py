import sys,copy
n=int(input("input n(n<=4): "))

training_file=open(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-train.word","r",encoding='utf-8')

def train(training_file,n):
    counts=dict()
    context_counts=dict()
    for line in training_file:
        line=line.strip()
        if len(line)!=0:
            sentence=line.split(" ")#スペースで単語を分ける
            sentence.append("</s>")#文末記号
            sentence.insert(0,"<s>")#文頭記号
            
            #每一行前n-1个词没有数,此处就会数前n-1个词
            #Todo:计算前n-1个词的集合的所有1-gram到(n-1)-gram的个数：for [1,2,3] count:1,2,3,12,23,123
            for word in sentence[0:n-1]:
                if word in counts:
                    counts[word]+=1
                else:
                    counts[word]=1
            
            #ngram从第n-1个开始取[0 ~ n-1]:
            for i in range(n-1,len(sentence)):
                tmp=n
                while tmp>=1:#算出第n-1个词的n-gram,(n-1)-gram...,1-gram
                    tmp_word=sentence[i-tmp+1:i+1]
                    word=" ".join(tmp_word)#w_i-n+1...w_i
                    #print(tmp_word)
                    tmp_word.pop()#remove the w_i to make the context
                    #print(tmp_word)
                    context_word=" ".join(tmp_word)#only context words 
                    #print(context_word)
                    if word in counts:
                        counts[word]+=1
                    else:
                        counts[word]=1
                    if context_word in context_counts:
                        context_counts[context_word]+=1
                    else:
                        context_counts[context_word]=1
                    tmp-=1#下一循环计算n-1-gram

    #add number of "" for 1-gram
    counts[""]=context_counts[""]                    
    training_file.close()

    total_count=copy.deepcopy(counts)#ngram
    for key,value in context_counts.items():
        if key in total_count:
            total_count[key]+=value
        else:
            total_count[key]=value
    
    ans=open("model_file_ngram.word","w",encoding='utf-8')
    ct=open("model_ngram_count.word","w",encoding='utf-8')
    for ngram in counts:
        ct.write(ngram+" "+str(counts[ngram])+"\n")#単語（文脈）の数をファイルに出力(Witten-Bell法)
        tmp=ngram.split(" ")
        tmp.pop(-1)
        context=" ".join(tmp)
        #print(context)
        prob=float(counts[ngram]/context_counts[context])
        ans.write(ngram+" "+str(prob)+"\n")
    ans.close()
    print("training finished")

train(training_file,n)

