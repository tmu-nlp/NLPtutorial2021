import sys

def trainning_model(path):
    emit,transition,context=dict(),dict(),dict()
    with open(path,'r',encoding='utf-8') as file:
        model=open("./model_file.txt","w",encoding='utf-8')
        for line in file:
            line=line.strip()
            previous="<s>"
            if previous in context:
                context[previous]+=1
            else:
                context[previous]=1
            wordtags=line.split(" ")
            for wordtag in wordtags:
                word,tag=wordtag.split("_")
                if previous+" "+tag in transition:
                    transition[previous+" "+tag]+=1
                else:
                    transition[previous+" "+tag]=1

                if tag in context:
                    context[tag]+=1
                else:
                    context[tag]=1

                if tag+" "+word in emit:
                    emit[tag+" "+word]+=1
                else:
                    emit[tag+" "+word]=1
                
                previous = tag
            if previous+" </s>" in transition:
                transition[previous+" </s>"]+=1
            else:
                transition[previous+" </s>"]=1
        
        for key,value in transition.items():
            #print(key+"-----"+str(value))
            previous,word =key.split(" ")
            #print("T "+key+" "+str(value/context[previous]))
            model.write("T "+key+" "+str(value/context[previous])+'\n')
        for key,value in emit.items():
            tag,word=key.split(" ")
            #print("E "+key+" "+str(value/context[tag]))
            model.write("E "+key+" "+str(value/context[tag])+'\n')
        model.close()
    print("trainning finished")
if __name__=='__main__':
    #trainning_model(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\test\05-train-input.txt")#test
    trainning_model(r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-train.norm_pos")