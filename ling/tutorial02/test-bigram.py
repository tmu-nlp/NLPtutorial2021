import sys,math

prob=dict()#確率を保存する辞書
# パラメータ
l1=0.95
l2=0.9
V=1000000.0 #未知語を含む語彙数
W=0
H=0.0
unk=0

read_model=open("./model_file_bigram.word","r",encoding='utf-8')#モデルを読み込む
for line in read_model:
    line=line.strip()
    line=line.split(" ")
    if len(line)==2:#1gram
        prob[line[0]]=line[1]
    else:#2gram
        prob[line[0]+" "+line[1]]=line[2]
read_model.close()

test_file=open(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.word","r",encoding='utf-8')
for line in test_file:
    sentence=line.strip().split(" ")
    sentence.append("</s>")
    sentence.insert(0,"<s>")
    for i in range(1,len(sentence)-1):
        word=sentence[i-1]+" "+sentence[i]
        if word in prob:
            p1=l1*float(prob[sentence[i]])+(1.0-l1)/V
            p2=l2*float(prob[word])+(1.0-l2)*p1
            H+=-math.log2(p2)
            W+=1
        else:
            unk+=1
print("entropy= "+ str(+H/W))
    
