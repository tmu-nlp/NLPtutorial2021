from collections import defaultdict
import numpy as np

def create_features(x,ids):
    phi=np.zeros(len(ids),dtype=np.float64)
    words=x.split(' ')
    for word in words:
        phi[ids["UNI"+word]]+=1
    return phi

def predict_one(w,phi):
    score=np.dot(w,phi)
    return (1 if score[0]>=0 else -1)

def forward_nn(network,phi0):
    phi= np.zeros(len(network))
    phi[0]=phi0
    for i in range(0,len(network)):
        w,b=network[i]
        phi[i]=np.tanh(np.dot(w,phi[i-1])+b)
        return phi

def backward_nn(network,phi,y_pred):
    J=len(network)
    delta=np.zeros(J+1)
    delta[-1]=np.array([y_pred-phi[J][0]])
    delta_d=np.zeros(J+1)
    for i in range(J-1,0,-1):
        delta_d[i+1]=delta[i+1]*(1-phi[i+1]**2)
        w,b=network[i]
        delta[i]=np.dot(delta_d[i+1],w)
    return delta_d

def update_weights(network,phi,delta_d,step):
    for i in range(0,len(network)):
        w,b=network[i]
        w+=step*np.outer(delta_d[i+1],phi[i])
        b+=step*delta_d[i+1]

def train(iteration,step,layer,node,input_file):
    ids=defaultdict(lambda:len(ids))
    feat_lab=[]
    with open(input_file,'r',encoding='utf-8') as f:
        for line in f:
            y,x=line.strip().split('\t')
            for word in x.split(' '):
                ids["UNI:"+word]
    with open(input_file,'r',encoding='utf-8') as f:
        for line in f:
            y,x=line.strip().split('\t')
            feat_lab.append((create_features(x,ids),y))
    

    #nn初期化
    net=[]
    #入力層
    w_in=np.random.rand(node,len(ids))-0.5
    b_in=np.random.rand(node)-0.5
    net.append((w_in,b_in))
    #隠れ層
    for _ in range(layer-1):
        w_mid=np.random.rand(node,node)-0.5
        b_mid=np.random.rand(node)-0.5
        net.append((w_mid,b_mid))
    #出力層
    w_out=np.random.rand(1,node)-0.5
    b_out=np.random.rand(1)-0.5
    
    #training
    for i in range(iteration):
        for phi0,y in feat_lab:
            phi=forward_nn(net,phi0)
            delta_d=backward_nn(net,phi,y)
            update_weights(net,phi,delta_d,step)

    return net,ids

if __name__=="__main__":
    input_file="/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-train.labeled"
    net,ids=train(1,0.001,1,2,input_file)
    print(net)
    
