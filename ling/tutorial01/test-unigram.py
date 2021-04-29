import sys,math

prob=dict()#確率を保存する辞書
# パラメータ
lambda1=0.95
lambda_unk=1.0-lambda1#未知語確率
V=1000000.0 #未知語を含む語彙数
W=0
H=0.0
unk=0

read_model=open("./model_file.word","r")#モデルを読み込む
for line in read_model:#各単語の確率を読み取って辞書に保存
    line=line.strip()
    line=line.split(" ")
    prob[line[0]]=line[1]
read_model.close()
#print(prob)

test_file=open("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/wiki-en-test.word","r")
for line in test_file:
    words=line.split(" ")
    words.append("</s>")
    for w in words:
        W+=1#単語数+1
        p=lambda_unk/V#未知語確率を先に加える（(1−λ1)*1/N）
        if w in prob:
            p+=lambda1*float(prob[w])#単語の確率を加える（λ1P_ML(wi)）
        else:
            unk+=1#未知語の数+1
        H+=-math.log2(p)#ｐの負の底２の対数を取る　−log2　P(w∣M)
print("entropy= "+str(H/W))#負の底２の対数尤度を単語数で割った値がエントロピー

print("coverage= "+str((W-unk)/W))#現れた単語の中で、モデルに含まれている割合
