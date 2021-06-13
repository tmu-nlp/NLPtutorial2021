from pickle import encode_long
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class RNN:
    '''
    word_ids : a dict stored the id of each word. key : word, value : id of the word
    '''
    def __init__(self,node):
        self.word_ids=defaultdict(lambda :len(self.word_ids))
        self.tag_ids=defaultdict(lambda:len(self.tag_ids))
        self.node=node
        self.net=[]
        self.feat_label=[]
        self.lr=0.01

    def init_net(self):
        w_r_x=(np.random.rand(self.node,len(self.word_ids))-0.5)/5
        w_r_h=(np.random.rand(self.node,self.node)-0.5)/5
        w_o_h=(np.random.rand(len(self.tag_ids),self.node)-0.5)/5
        b_r=(np.random.rand(self.node)-0.5)/5
        b_o=(np.random.rand(len(self.tag_ids))-0.5)/5
        self.net=[w_r_x,w_r_h,w_o_h,b_r,b_o]

    def create_one_hot(self,id,size):
        vec=np.zeros(size)
        vec[id]=1
        return vec

    #return an index
    def find_best(self,p):
        y=0
        for i in range(len(p)):
            if p[i]>p[y]:
                y=i
        return y

    '''
    preprocessing of the train_file, get each word and pos of it, store them into list or dict
    
    @para
    train_file
    @return
    '''
    def preprocessing(self,train_file):
        with open(train_file,'r',encoding='utf-8') as f:
            for line in f:
                word_tag=line.strip().split(' ')
                sent=[]
                for x in word_tag:
                    word,tag=x.split("_")
                    self.word_ids[word]
                    self.tag_ids[tag]
                    sent.append((word,tag))
                self.feat_label.append(sent)
                #sent looks like[(word1,tag1),(word2,tag2)...]

    def forward_rnn(self,sent):
        w_r_x,w_r_h,w_o_h,b_r,b_o=self.net
        h=[]
        p=[]
        y=[]

        for t in range(len(sent)):
            x,_=sent[t]

            #print(str(self.word_ids[x])+' '+str(len(self.word_ids)))
            
            x=self.create_one_hot(self.word_ids[x],len(self.word_ids))

            if t>0:
                h.append(np.tanh(np.dot(w_r_x,x)+np.dot(w_r_h,h[t-1])+b_r))
            else:
                h.append(np.tanh(np.dot(w_r_x,x)+b_r))
            p.append(np.tanh(np.dot(w_o_h,h[t])+b_o))
            y.append(self.find_best(p[t]))
        
        return h,p,y

    def forward_rnn_test(self,words):
        w_r_x,w_r_h,w_o_h,b_r,b_o=self.net
        h=[]
        p=[]
        y=[]

        for t in range(len(words)):
            x=words[t]
            if x in self.word_ids:
                x=self.create_one_hot(self.word_ids[x],len(self.word_ids))
            else:
                x=np.zeros(len(self.word_ids))
            if t>0:
                h.append(np.tanh(np.dot(w_r_x,x)+np.dot(w_r_h,h[t-1])+b_r))
            else:
                h.append(np.tanh(np.dot(w_r_x,x)+b_r))
            p.append(np.tanh(np.dot(w_o_h,h[t])+b_o))
            y.append(self.find_best(p[t]))
        
        return h,p,y

    '''
    calculate the gradient of each parameter
    @para
    sent : a sentence looks like [(word1,tag1),(word2,tag2)...]
    h,p : parameters calculated by forward_rnn()
    @return
    [d_w_r_x,d_w_r_h,d_w_o_h,d_b_r,d_b_o] : a list of gradients
    '''
    def grad_rnn(self,sent,h,p):
        w_r_x,w_r_h,w_o_h,b_r,b_o=self.net

        d_w_r_x=np.zeros((self.node,len(self.word_ids)))
        d_w_r_h=np.zeros((self.node,self.node))
        d_b_r=np.zeros(self.node)
        d_w_o_h=np.zeros((len(self.tag_ids),self.node))
        d_b_o=np.zeros(len(self.tag_ids))
        d_r_p=np.zeros(self.node)

        for t in range(len(sent)-1,-1,-1):
            word,tag=sent[t]
            p_p=self.create_one_hot(self.tag_ids[tag],len(self.tag_ids))
            x=self.create_one_hot(self.word_ids[word],len(self.word_ids))

            d_o_p=p_p-p[t]
            d_w_o_h+=np.outer(h[t],d_o_p).T
            d_b_o+=d_o_p
            d_r=np.dot(d_r_p,w_r_h)+np.dot(d_o_p,w_o_h)
            d_r_p=d_r*(1-h[t]**2)
            d_w_r_x+=np.outer(x,d_r_p).T
            d_b_r+=d_r_p

            if t!=0 :
                d_w_r_h+=np.outer(h[t-1],d_r_p).T
        return [d_w_r_x,d_w_r_h,d_w_o_h,d_b_r,d_b_o]
    
    '''
    The process of updating weight,recieve weights and bias from network and delta the function grad_rnn(),
    add the learning rate lr. After the process the parameters in network will be updated

    @para
    delta: the return value from function grad_rnn()
    lr: learning rate
    @return 

    '''
    def update_weight(self,delta,lr):
        w_r_x,w_r_h,w_o_h,b_r,b_o=self.net
        d_w_r_x,d_w_r_h,d_w_o_h,d_b_r,d_b_o=delta

        w_r_x+=lr*d_w_r_x
        w_r_h+=lr*d_w_r_h
        w_o_h+=lr*d_w_o_h
        b_r+=lr*d_b_r
        b_o+=lr*d_b_o

        self.net=[w_r_x,w_r_h,w_o_h,b_r,b_o]
    '''
    train the model,recieve the path of train file, number of iteration(epochs)
    firstly preprocess the file,initiate the whole network
    as the training goes, the hyperparameter will be updated

    @para
    train_file
    iter:epochs

    @return

    '''
    def train(self,train_file,iter):
        self.preprocessing(train_file)
        self.init_net()
        for _ in tqdm(range(iter)):
            for sent in self.feat_label:
                h,p,y=self.forward_rnn(sent)
                delta=self.grad_rnn(sent,h,p)
                self.update_weight(delta,self.lr)

    '''
    recieve test file,output each pos of the word in test file
    
    @para:
    test_file
    ans_file
    
    @return
    
    '''
    def predict(self,test_file,ans_file):
        with open(test_file,'r',encoding='utf-8') as f,open(ans_file,'w',encoding='utf-8') as a:
            for line in f:
                pos=[]
                line=line.strip().split(' ')
                h,p,y=self.forward_rnn_test(line)
                #print(y)
                for e in y:
                    for key,value in self.tag_ids.items():
                        if e ==value:
                            pos.append(key)
                a.write(" ".join(pos)+'\n')
    

rnn=RNN(64)
#test
'''
rnn.train("/Users/lingzhidong/Documents/GitHub/nlptutorial/test/05-train-input.txt",40)
rnn.predict("/Users/lingzhidong/Documents/GitHub/nlptutorial/test/05-test-input.txt","testans.txt")
'''
#enshu

rnn.train("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/wiki-en-train.norm_pos",5)
rnn.predict("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/wiki-en-test.norm","ans.txt")

'''
epoch=5,node=64
Accuracy: 85.36% (3895/4563)

Most common mistakes:
JJ --> NN       105
NNS --> NN      82
NNP --> NN      66
VBN --> NN      43
RB --> NN       41
VBG --> NN      33
VBP --> NN      29
CD --> NN       26
VB --> NN       18
IN --> WDT      17

'''