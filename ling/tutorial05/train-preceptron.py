import sys
from typing import Collection
from collections import defaultdict

def load_file(model_file):
    w=defaultdict(int)
    with open(model_file,'r') as f:
        for line in f:
            prob=line.strip().split(' ')
            w["UNI:"+prob[0]]=float(prob[1])
    return w

def predict_one(w,phi):
    score=0
    for name,value in phi.items():
        if name in w:
            score+=value*w[name]
    if score>=0:
        return 1
    else:
        return -1

def create_features(x):
    phi=defaultdict(int)
    words=x.strip().split()
    for word in words:
        phi["UNI:"+word]+=1
    return phi

def predict_all(w,input_file):
    y_p=[]      
    with open(input_file,'r') as i:
        for line in i:
            if '\t' in line:
                l,line=line.split('\t')
            #print(line)
            phi=create_features(line.strip())
            #print(phi.items())
            y_=predict_one(w,phi)
            #print(y)
            y_p.append(y_)
    return y_p

def update_weights(w,phi,y):
    for name,value in phi.items():
        w[name]+=float(value*y)

def online_train(i,input_file):
    w=defaultdict(int)
    with open(input_file,'r') as f :
        for a in range(0,i):
            for line in f:
                #print(line)
                y,x=line.strip().split('\t')
                #print(x)
                y=int(y)
                phi=create_features(x)
                y_=predict_one(w,phi)
                if y_!=y:
                    update_weights(w,phi,y)
    return w



if  __name__ == "__main__":
    model_file="/Users/lingzhidong/Documents/GitHub/NLPtutorial2021/ling/tutorial01/model_file.word"
    input_file="/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-test.word"
    train_file="/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-train.labeled"
    w=online_train(10,train_file)
    a=predict_all(w,input_file)
    ans=open('ans.labeled','w')
    for x in a:
        ans.write(str(x)+'\n')
    ans.close()

'''
Accuracy = 90.967056%
'''