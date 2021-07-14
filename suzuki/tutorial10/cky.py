from collections import defaultdict
import math


def PRINT(sym_i_j, best_edge, words):
    sym, i, j = sym_i_j.split(' ')
    if sym_i_j in best_edge.keys():
        return '(' + sym + ' ' + PRINT(best_edge[sym_i_j][0], best_edge, words) + ' ' + PRINT(best_edge[sym_i_j][1], best_edge, words) + ')'
    else:
        return '(' + sym + ' ' + words[int(i)] + ')'

if __name__ == '__main__':
    #文法の読み込み
    nonterm = []
    preterm = defaultdict(lambda: [])
    with open('wiki-en-test.grammar', 'r') as f:
        for rule in f:
            lhs, rhs, prob = rule.strip().split('\t')
            rhs_symbols = rhs.split(' ')
            if len(rhs_symbols) == 1:
                preterm[rhs].append((lhs, math.log(float(prob))))
            else:
                nonterm.append((lhs, rhs_symbols[0], rhs_symbols[1], math.log(float(prob))))
    
    with open('tutorial10.ans', 'w') as ans:
        with open('wiki-en-short.tok', 'r') as f:
            for line in f:
                #前終端記号を追加
                words = line.strip().split(' ')
                best_score = defaultdict(lambda: -math.inf)
                best_edge = {}
                for i, word in enumerate(words):
                    for lhs, log_prob in preterm[word]:
                        best_score['{} {} {}'.format(lhs, i, i+1)] = log_prob
    
                #非終端記号の組み合わせ
                for j in range(2, len(words)+1):
                    for i in range(j-2, -1, -1):
                        for k in range(i+1, j):
                            for sym, lsym, rsym, logprob in nonterm:
                                if best_score['{} {} {}'.format(lsym, i, k)] > -math.inf and best_score['{} {} {}'.format(rsym, k, j)] > -math.inf:
                                    my_lp = best_score['{} {} {}'.format(lsym, i, k)] + best_score['{} {} {}'.format(rsym, k, j)] + logprob
                                    if my_lp > best_score['{} {} {}'.format(sym, i, j)]:
                                        best_score['{} {} {}'.format(sym, i, j)] = my_lp
                                        best_edge['{} {} {}'.format(sym, i, j)] = ('{} {} {}'.format(lsym, i, k), '{} {} {}'.format(rsym, k, j))

                #木を出力
                ans.write(PRINT('S 0 {}'.format(len(words)), best_edge, words))
                ans.write('\n')