import sys,math
prob=dict()

lambda1=0.95
lambda_unk=1.0-lambda1
V=1000000.0
W=0
H=0.0
unk=0

read_model=open("./model_file.word","r")
for line in read_model:
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
        W+=1
        p=lambda_unk/V
        if w in prob:
            p+=lambda1*float(prob[w])
        else:
            unk+=1
        H+=-math.log2(p)
print("entropy= "+str(H/W))
print("coverage= "+str((W-unk)/W))