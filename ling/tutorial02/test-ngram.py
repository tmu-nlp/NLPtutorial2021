import sys,math

#線形補間を用いた確率
def cal_prob_linear(sentence,n,prob,Lambda):
    if n>0:
        cur_word=" ".join(sentence[-n:])
        n-=1
        if n!=0:
            p=Lambda[n-1]*float(prob[cur_word])+(1-Lambda[n-1])*cal_prob_linear(sentence,n,prob,Lambda)
        else:
            p=Lambda[n-1]*float(prob[cur_word])+(1-Lambda[n-1])/V
        return p
#witten bell smoothingを用いたλ
def cal_prob_witten_bell(sentence,n,prob,count):
    if n>0:
        cur_word_list=sentence[-n:]
        cur_word=" ".join(sentence[-n:])
        uniq_after_w_i=0
        context=" ".join(cur_word_list[:-1])
        for key in count:
            if (context in key) & (len(context)<len(key)):
                uniq_after_w_i+=1
        lambda_w_i_1=1.0-uniq_after_w_i/(uniq_after_w_i+float(count[str(context)]))
        n-=1
        if n!=0:
            p=lambda_w_i_1*float(prob[cur_word])+(1.0-lambda_w_i_1)*cal_prob_witten_bell(sentence,n,prob,count)
        else:
            p=lambda_w_i_1*float(prob[cur_word])+(1.0-lambda_w_i_1)/V
        return p

prob=dict()#確率を保存する辞書
count=dict()#単語の数を保存する辞書
# パラメータ
Lambda=[0.95,0.9,0.9]
V=1000000.0 #未知語を含む語彙数
W=0
H=0.0
unk=0
n=int(input("input n=n in train: "))

#モデルを読み込む
read_model=open("./model_file_ngram.word","r",encoding='utf-8')
for line in read_model:
    line=line.strip()
    word=line.split(" ")
    prob[" ".join(word[:-1])]=word[-1]
read_model.close()

#単語の数を読み込む
read_count=open("./model_ngram_count.word","r",encoding='utf-8')
for line in read_count:
    line=line.strip()
    word=line.split(" ")
    count[" ".join(word[:-1])]=word[-1]
read_count.close()

test_file=open(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.word","r",encoding='utf-8')
for line in test_file:
    sentence=line.strip().split(" ")
    sentence.append("</s>")
    sentence.insert(0,"<s>")
    for i in range(n-1,len(sentence)-1):
        word=" ".join(sentence[i-n+1:i+1])
        W+=1
        if word in prob:
            #p=cal_prob_linear(sentence[i-n+1:i+1],n,prob,Lambda)
            p=cal_prob_witten_bell(sentence[i-n+1:i+1],n,prob,count)
            H+=-math.log2(p)             
        else:
            unk+=1
print("entropy= "+ str(+H/W))
    
