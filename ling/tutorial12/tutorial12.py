from collections import defaultdict
import subprocess
from tqdm import tqdm

def create_features(X,Y):
    phi=defaultdict(int)
    for i,pos in enumerate(range(Y)+1):
        if i==0:
            first_tag="<s>"
        else:
            first_tag=Y[i-1]
        if i==len(Y):
            next_tag="</s>"
        else:
            next_tag=Y[i]

        phi[f"T {first_tag} {next_tag}"]+=1
    for i in range(len(Y)):
        phi[f"E {Y[i]} {X[i]}"]+=1
        if caps(X[i]):
            phi[f'CAPS {X[i]}']+=1
    return phi

def HMM_viterbi(words,w,possible_tags,trans):
    l = len(words)

    best_score = {'0 <s>': 0}
    best_edge = {'0 <s>': None}

    for i in range(l):
        for prev in possible_tags:
            key = f'{i} {prev}'
            for nxt in possible_tags:
                t_key, e_key = f'{prev} {nxt}', f'E {nxt} {words[i]}'
                if key not in best_score or t_key not in trans:
                    continue
                score = best_score[key] + w[f'T {t_key}'] + w[e_key]
                n_key = f'{i+1} {nxt}'
                if n_key not in best_score or best_score[n_key] < score:
                    best_score[n_key] = score
                    best_edge[n_key] = key

    for tag in possible_tags:
        if not trans[f'{tag} </s>']:
            continue

        key, t_key = f'{l} {tag}', f'{tag} </s>'
        score = best_score[key] + w[f'T {t_key}']
        n_key = f'{l+1} </s>'
        if n_key not in best_score or best_score[n_key] < score:
            best_score[n_key], best_edge[n_key] = score, key
    
    tags, next_edge = [], best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        _, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags = tags[::-1]

    return tags
    
def update_weight(w,phi_prime,phi_hat):
    for key,value in phi_prime.items():
        w[k]+=value
    for k,v in phi_hat.items():
        w[k]-=v

def caps(word):
    return True if word[0].isupper() else False    

def preprocess(words_tags):
    trans = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for (_, tags) in words_tags:
        prevs, nexts = ['<s>'] + tags, tags + ['</s>']
        for prv, Next in zip(prevs, nexts):
            trans[f'{prv} {Next}'] += 1
        possible_tags.update(tags)

    return trans, possible_tags


'''
output:[[word1,pos1],[word2,pos2]...]
'''
def import_trainfile(path):

    with open(path,'r',encoding='utf-8') as f:
        data=f.readlines()
    sent_tags=[]
    for line in data:
        line=line.strip().split(' ')
        words,tags=[],[]
        for word_tag in line:
            #print(word_tag)
            word,tag=word_tag.split('_')
            words.append(word)
            tags.append(tag)
        sent_tags.append((words,tags))
    #print(sent_tags[0])
    return sent_tags
    
def import_testfile(path):
    with open(path,'r',encoding='utf-8') as f:
        data=f.readlines()
    sentences=[]
    for line in data:
        words=line.strip().split(' ')
        sentences.append(words)
    return sentences

def train_HMM_perceptron(words_tags,w,possible_tags,trans):
    for (words,tags_prime) in words_tags:
        tags_hat=HMM_viterbi(w,words,possible_tags,trans)
        phi_prime=create_features(words,tags_prime)
        phi_hat=create_features(words,tags_hat)

        update_weight(w,phi_prime,phi_hat)
    
if __name__=="__main__":
    w=defaultdict(lambda:0)
    
    trainfile=r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-train.norm_pos"
    testfile=r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.norm"
    words_tags=import_trainfile(trainfile)
    sentences=import_testfile(testfile)
    trans,possible_tags=preprocess(words_tags)
    for _ in tqdm(range(20)):
        train_HMM_perceptron(words_tags,w,possible_tags,trans)
    
    with open('weights.txt','w',encoding='utf-8') as f_o:
        for feat , weight in w.items():
            f_o.write(f'{feat}{weight}\n')

    with open('my_result.txt','w',encoding='utf-8') as f_o:
        for words in sentences:
            tags=HMM_viterbi(w,words,possible_tags,trans)
            f_o.write(''.join(tags)+'\n')
    
    script_path = r'C:\Users\Lexus\Documents\GitHub\Nlptutorial\script\gradepos.pl'
    test_path=r"C:\Users\Lexus\Documents\GitHub\Nlptutorial\data\wiki-en-test.pos"
    out_path=r"C:\Users\Lexus\Documents\GitHub\NLPtutorial2021\ling\tutorial12\my_result.txt"
    subprocess.run(f'python2 {script_path} {test_path} {out_path}'.split())