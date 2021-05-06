import sys,math

def cal_prob(sentence,n,Lambda):
    if n>0:
        cur_word=" ".join(sentence[-n:])
        n-=1
        if n!=1:
            p=Lambda[n-1]*float(prob[cur_word])+(1-Lambda[n-1])*cal_prob(sentence,n,Lambda)
        else:
            p=Lambda[n-1]*float(prob[cur_word])+(1-Lambda[n-1])/V
        return p

prob=dict()#確率を保存する辞書
# パラメータ
Lambda=[0.95,0.9,0.9]
V=1000000.0 #未知語を含む語彙数
W=0
H=0.0
unk=0
n=int(input("input n=n in train: "))

read_model=open("./model_file_ngram.word","r",encoding='utf-8')#モデルを読み込む
for line in read_model:
    line=line.strip()
    word=line.split(" ")
    prob[" ".join(word[:-1])]=word[-1]
read_model.close()

#print(prob["processing -LRB- NLP"])
print(str(cal_prob(["processing -LRB- NLP"],n,Lambda)))
test_file=open(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.word","r",encoding='utf-8')
for line in test_file:
    sentence=line.strip().split(" ")
    sentence.append("</s>")
    sentence.insert(0,"<s>")
    for i in range(n-1,len(sentence)-1):
        word=" ".join(sentence[i-n+1:i+1])
        if word in prob:
            for j in range(1,n+1):
                H+=-math.log2(cal_prob(sentence[i-n+1:i+1],n,Lambda))
                W+=1
        else:
            unk+=1
print("entropy= "+ str(+H/W))
    
