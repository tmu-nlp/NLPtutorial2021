import sys,math
from typing import Collection
from collections import defaultdict

def create_features(x):
    phi=defaultdict(int)
    words=x.strip().split()
    for word in words:
        phi["UNI:"+word]+=1
    return phi

def predict_one(w,phi):
    score=0
    for name,value in phi.items():
        if name in w:
            score+=value*w[name]
    return sign(score)
    #return sigmoid(score)

'''
args:float
output: args>=0 => 1 ; args<0 => -1
'''
def sign(score):
    if score>=0:
        return 1
    else:
        return -1

'''
args: float
output: float
'''
def sigmoid(score):
    return math.pow(math.e,score)/(1+math.pow(math.e,score))


def predict_all(w,input_file):
    y_p=[]      
    with open(input_file,'r',encoding='utf-8') as i:
        for line in i:
            if '\t' in line:
                l,line=line.split('\t')
            phi=create_features(line.strip())
            y_=predict_one(w,phi)
            y_p.append(y_)
    return y_p

#new
def update_weights(w,phi,y,c):
    for name,value in w.items():
        if abs(value)<c:
            w[name]=0
        else:
            w[name]-=sign(value)*c
    for name,value in phi.items():
        w[name]+=value*y


#new
'''
args: dict A, dict B
output: sum(A[x]*B[x]) , x in A & B
'''
def dot_w_phi(w,phi):
    sum=0
    for key,value in phi.items():
        sum+=w[key]*phi[key]
    return sum

#new
'''
args: i (iteration times(epoch)) ,train file path, margin, c
output: dict w (trained weight)
'''
def online_train(i,input_file,margin,c):
    w=defaultdict(int)
    with open(input_file,'r',encoding='utf-8') as f :
        for a in range(0,i):
            for line in f:
                y,x=line.strip().split('\t')
                y=int(y)
                phi=create_features(x)
                val=y*dot_w_phi(w,phi)
                #y_=predict_one(w,phi)
                if val<=margin:
                    update_weights(w,phi,y,c)
    return w

def getw(w,name,c,iter,last):
    if iter!=last[name]:
        c_size=c*(iter-last[name])
        if abs(w[name])<=c_size:
            w[name]=0
        else:
            w[name]-=sign(w[name])*c_size
        last[name]=iter
    return w[name]


if  __name__ == "__main__":
    input_file="/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-test.word"
    train_file="/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-train.labeled"
    #input_file=r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\titles-en-test.word"
    #train_file=r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\titles-en-train.labeled"
    margin=5.0
    c=0.0001
    w=online_train(20,train_file,margin,c)
    a=predict_all(w,input_file)
    ans=open('ans.labeled','w')
    for x in a:
        ans.write(str(x)+'\n')
    ans.close()
    print("Training finished")
    
'''
margin=1.0;c=0.0001
Accuracy = 91.285866%

margin=2.0;c=0.0001
Accuracy = 91.108750%

margin=3.0;c=0.0001
Accuracy = 91.640099%

margin=4.0;c=0.0001
Accuracy = 89.585547%

margin=5.0;c=0.0001
Accuracy = 91.179596%
'''