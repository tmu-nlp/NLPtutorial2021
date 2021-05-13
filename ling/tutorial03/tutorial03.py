import math,sys

def read_model(path):
    prob=dict()
    with open(path,'r',encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            word=line.split("\t")
            prob[" ".join(word[:-1])]=word[-1]
    return prob

def word_segment(model_file,text_file,ans_file):
    #load model
    model=read_model(model_file)
    #read inputs
    with open(text_file,'r',encoding='utf-8') as t:
        inputs=t.readlines()
    #open an file for recording answer
    ans=open(ans_file,'w',encoding='utf-8')
    
    for line in inputs:
        line.strip()
        best_edge=dict()
        best_score=dict()
        best_edge[0]=None
        best_score[0]=0
        
        #forward step
        for word_end in range(1,len(line)+1):
            best_score[word_end]=math.inf
            for word_begin in range(0,word_end):
                word=line[word_begin:word_end]
                if word in model or len(word)==1:
                    prob=0.95*float(model[word])+0.05*1000000.0
                    my_score=best_score[word_begin]-math.log2(prob)
                    if my_score<best_score[word_end]:
                        best_score[word_end]=my_score
                        best_edge[word_end]=[word_begin,word_end]
    
        #backward step
        words=[]
        next_edge=best_edge[len(best_edge)-1]
        while next_edge!=None:
            word=line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge=best_edge[next_edge[0]]
        words.reverse()
        ans.write(" ".join(words)+"\n")
    ans.close()

if __name__=="__main__":
    model_path=r'C:\Users\Lexus\Documents\GitHub\Nlptutorial\test\04-model.txt'
    text_path=r'C:\Users\Lexus\Documents\GitHub\Nlptutorial\test\04-input.txt'
    ans_path='test_ans.txt'
    #model=read_model(model_path)
    #print(model)
    word_segment(model_path,text_path,ans_path)