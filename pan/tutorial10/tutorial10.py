from math import log, inf
from collections import defaultdict

nonterm = []    #(左, 右1, 右2, 確率)
preterm = defaultdict(list)    #pre[右] = [(左, 確率)]
best_edge  = {}

def loade_grammer(grammer_file):
    global nonterm
    global preterm
    with open(grammer_file, encoding = 'utf-8') as f:
        for line in f:
            lhs, rhs, prob = line.split('\t')
            prob = float(prob)
            rhs = rhs.split()
            if len(rhs) == 1:
                preterm[rhs[0]].append([lhs, log(prob)])
            else:
                nonterm.append([lhs, rhs[0], rhs[1], log(prob)])

def CKY(sentence, ans_file):
    global nonterm
    global preterm
    best_score = defaultdict(lambda: -inf)
    global best_edge
    best_edge = {}

    words = sentence.strip().split()
    for i in range(len(words)):
        if words[i] in preterm:
            for lhs, log_prob in preterm[words[i]]:
                best_score[f'{lhs}|{i}|{i+1}'] = log_prob

    for j in range(2, len(words)+1):
        for i in range(j - 1)[::-1]:
            for k in range(i+1, j):
                for sym, lsym, rsym, logprob in nonterm:
                    if f'{lsym}|{i}|{k}' in best_score and f'{rsym}|{k}|{j}' in best_score:
                        my_lp = best_score[f'{lsym}|{i}|{k}'] + best_score[f'{rsym}|{k}|{j}'] + logprob
                        if my_lp > best_score[f'{sym}|{i}|{j}']:
                            best_score[f'{sym}|{i}|{j}'] = my_lp
                            best_edge[f'{sym}|{i}|{j}'] = (f'{lsym}|{i}|{k}', f'{rsym}|{k}|{j}')
    with open(ans_file, 'a') as f:
        f.write(PRINT(f'S|0|{len(words)}', words))

def PRINT(sym_ij, words):
    global best_edge
    sym, i, _ = sym_ij.split('|')
    if sym_ij in best_edge:
        return f'({sym} {PRINT(best_edge[sym_ij][0], words)} {PRINT(best_edge[sym_ij][1], words)})'
    else:
        i = int(i)
        return f'({sym} {words[i]})'

if __name__ == '__main__':
    gram = '/users/kcnco/github/NLPtutorial2021/pan/tutorial10/wiki-en-test.grammar'
    inp = 'users/kcnco/github/NLPtutorial2021/pan/tutorial10/wiki-en-short.tok'
    ans = 'users/kcnco/github/NLPtutorial2021/pan/tutorial10/my_ans'
    loade_grammer(gram)
    with open(ans, 'w') as f:
        pass
    with open(inp, encoding = 'utf-8') as f:
        for line in f:
            CKY(line, ans)
