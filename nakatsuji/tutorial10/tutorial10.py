from collections import defaultdict
from nltk.tree import Tree
from nltk.treeprettyprinter import TreePrettyPrinter
import math
path = './nlptutorial/'

#文法の読み込み
nonterm = []
preterm = defaultdict(list)
with open(path + 'test/08-grammar.txt') as grammar_file:
    for rule in grammar_file:
        lhs, rhs, prob = rule.strip().split('\t')
        rhs_symbols = rhs.split()
        if len(rhs_symbols) == 1:#全終端記号
            preterm[rhs].append((lhs, math.log(float(prob))))
        else:#非終端記号
            nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))))


def PRINT(symij, words, best_edge):
    sym, i, j = symij.split(":")
    if symij in best_edge:
        return f'({sym} {PRINT(best_edge[symij][0], words, best_edge)} {PRINT(best_edge[symij][1], words, best_edge)})'
    else:#終端記号
        return f'({sym} {words[int(i)]})'
    


#parse
with open(path + 'test/08-input.txt') as f, open('./tutorial10/w_my_ans.txt', 'w') as ans:
    for line in f:
        words = line.split()
        m_INF = -float('inf')
        best_score = defaultdict(lambda: m_INF)
        best_edge = {}
        #前終端記号を追加
        for i in range(len(words)):
            if preterm[words[i]] != []:
                for lhs, log_prob in preterm[words[i]]:
                    best_score[f'{lhs}:{i}:{i+1}'] = log_prob
        #非終端記号の組み合わせ
        for j in range(2, len(words)+1):
            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    #各文法ルールを展開
                    for sym, lsym, rsym, logprob in nonterm:
                        now = f'{sym}:{i}:{j}'  #key
                        l = f'{lsym}:{i}:{k}'   #key
                        r = f'{rsym}:{k}:{j}'   #key
                        if best_score[l] > m_INF and best_score[r] > m_INF:
                            my_lp = best_score[l] + best_score[r]
                            if my_lp > best_score[now]:
                                best_score[now] = my_lp
                                best_edge[now] = \
                                    (f'{l}', f'{r}')
        print(PRINT(f'S:0:{len(words)}', words, best_edge), file=ans)

        #visual
        tree_line = PRINT(f'S:0:{len(words)}', words, best_edge)
        t = Tree.fromstring(tree_line)
        print(TreePrettyPrinter(t).text())
        print(tree_line)