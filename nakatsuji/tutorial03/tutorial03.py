###viterbi###
import sys, os, math
from collections import *
from train01 import prob, train_unigram

def words_seg_viterbi(edges_dic, input_file, write_ans):
    #model = open(model_file, encoding='uft-8')
    #input = open(text_file, encoding='utf-8')
    #answer = open(ans_file, encoding='utf-8')

    
    
    with open(input_file, encoding='utf-8') as f:
        lines = f.readlines()
    
    with open(write_ans, 'w',encoding='utf-8') as f:
        Lamda = 0.95
        V = 10 ** 6
        for line in lines:
            line.strip()
            #前向きステップ
            best_edge = defaultdict(lambda:0)
            best_score = defaultdict(lambda:0)
            best_edge[0] = None
            best_score[0] = 0
            
            for word_end in range(1, len(line)+1):
                best_score[word_end] = 10 ** 10
                for word_begin in range(word_end):
                    word = line[word_begin: word_end]
                    if word in edges_dic or len(word) ==  1:
                        prob =  (1 - Lamda) / V
                        if word in edges_dic:
                            prob += Lamda * float(edges_dic[word])
                        my_score = best_score[word_begin] - math.log(prob)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = [word_begin, word_end]
            
            #後ろ向きステップ
            words = []
            next_edge = best_edge[len(best_edge) - 1]

            while next_edge != None:
                word = line[next_edge[0]: next_edge[1]]
                words.append(word)
                next_edge = best_edge[next_edge[0]]
            
            words.reverse()
            f.write(' '.join(words))
            #print(" ".join(words), end = '')



if __name__ == "__main__":
    path = '/Users/michitaka/lab/NLP_tutorial/nlptutorial/'
    #テスト
    test_model = path + 'test/04-model.txt'
    test_input = path + 'test/04-input.txt'
    test_answer = path + 'test/04-ans.txt'
    #テストで使うモデルを辞書に
    edges_dic = defaultdict(int)
    with open(test_model, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            edges_dic[line[0]] = line[1]

    words_seg_viterbi(edges_dic, test_input, 'my_ans.txt')
    



    #using data wiki
    wiki_model = path + 'data/wiki-ja-train.word'
    wiki_input = path + 'data/wiki-ja-test.txt'

    counts, total_counts = train_unigram(wiki_model)
    prob_dic = prob(counts, total_counts)
    words_seg_viterbi(edges_dic, wiki_input, 'ans_wiki.txt')



#低くね?
'''
Sent Accuracy: 0.00% (/84)
Word Prec: 46.02% (428/930)
Word Rec: 66.46% (428/644)
F-meas: 54.38%
Bound Accuracy: 66.19% (560/846)
'''