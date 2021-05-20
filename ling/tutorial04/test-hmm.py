import sys,math

emission,transition,possible_tags=dict(),dict(),dict()

def read_model(file):
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            Type,context,word,prob=line.split(" ")
            possible_tags[context]=1
            if Type=="T":
                transition[context+" "+word]=prob#transition[prev next]=>P_T(y_i|y_i-1)
            else:
                emission[context+" "+word]=prob#emission[next word_i]=>P_E(x_i|y_i)
                

def test(path):
    with open(path,'r',encoding='utf-8') as f:
        ans=open("./answer.txt",'w',encoding='utf-8')
        for line in f:
            words=line.strip().split(" ")
            l=len(words)
            best_score,best_edge=dict(),dict()
            best_score["0 <s>"]=0
            best_edge["0 <s>"]=None
            for i in range(0,l):
                for prev in possible_tags.keys():
                    for Next in possible_tags.keys():
                        if str(i)+" "+prev in best_score and prev+" "+Next in transition:
                            Lambda=0.95
                            N=1000000.0
                            P_E=(1.0-Lambda)/N
                            if Next+" "+words[i] in emission:
                                P_E+=Lambda*float(emission[Next+" "+words[i]])
                            
                            score=best_score[str(i)+" "+prev]-math.log2(float(transition[prev+" "+Next]))-math.log2(P_E)
                            
                            if str(i+1)+" "+Next not in best_score or best_score[str(i+1)+" "+Next]>score:
                                best_score[str(i+1)+" "+Next]=score
                                best_edge[str(i+1)+" "+Next]=str(i)+" "+prev
            #end of sentence
            for prev in possible_tags.keys():
                Next="</s>"

                if str(l)+" "+prev in best_score and prev+" "+Next in transition:
                        score=best_score[str(l)+" "+prev]-math.log2(float(transition[prev+" "+Next]))

                        if str(l+1)+" "+Next not in best_score or best_score[str(l+1)+" "+Next]>score:
                            best_score[str(l+1)+" "+Next]=score
                            best_edge[str(l+1)+" "+Next]=str(l)+" "+prev

            tags=[]
            next_edge=best_edge[str(l+1)+" </s>"]
            while next_edge !="0 <s>":
                position,tag=next_edge.split(" ")
                tags.append(tag)
                next_edge=best_edge[next_edge]
            tags.reverse()
            #print(" ".join(tags))
            ans.write(" ".join(tags)+"\n")
        ans.close()
        print("test finished")
if __name__=='__main__':
    read_model("./model_file.txt")
    #print(possible_tags.items())    #possible_tags[tag]=number
    #print(transition.items())       #transition["tag tag"]=prob
    #print(emission.items())         #emission["tag word"]=prob
    #test(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\test\05-test-input.txt")#test
    test(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.norm")