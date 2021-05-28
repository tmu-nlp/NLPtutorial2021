#1-gram
import sys
training_file=open("/Users/lingzhidong/Documents/GitHub/nlptutorial/test/03-train-input.txt","r")
counts=dict()#単語を数えるための辞書
total_count=0#単語の総数
for line in training_file:
    line=line.strip()#改行などを削除
    if len(line)!=0:
        if '\t' in line:
            label,sentence=line.split("\t")
        else:
            sentence=line
        sentence=line.split()
        sentence.append("</s>")#文末記号を加える
        for word in sentence:
            if word in counts:#辞書にある単語の場合　その単語の数+1　総数+1
                counts[word]+=1
                total_count+=1
            else:#辞書にない場合　単語を1から数える　総数+1
                counts[word]=1
                total_count+=1
training_file.close()

ans=open("model_file.word","w")
for word in counts:
    p=counts[word]/total_count#各単語の確率P(wi)を計算
    ans.write(str(word)+" "+str(p)+"\n")#単語とその確率を一行として書き込む
ans.close()

