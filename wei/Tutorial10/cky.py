import sys
import math
from collections import defaultdict



# p65:文法の読み込み
def load_grammar(filepath):
    nonterm = []    # 非終端記号[(lhs, rhs[0], rhs[1], prob)]
    preterm = defaultdict(lambda :[])
    with open(filepath, 'r' ,encoding='utf-8') as f:
        for rule in f:
            lhs, rhs, prob = rule.strip().split('\t')
            rhs_symbol = rhs.split()
            if len(rhs_symbol) == 1:    # 前終端記号
                # ['saw']:[('VBD',log(0.4)),...]
                preterm[rhs].append((lhs, math.log(float(prob))))
            else:
                # [(S,NP,VP,log(0.6)),...]
                nonterm.append((lhs, rhs_symbol[0], rhs_symbol[1], math.log(float(prob))))
    return nonterm, preterm

def predict_s_tree(words, grammar):
    nonterm, preterm = grammar
    best_score = {}
    best_edge = {}

    #words = line.split()
    len_wds = len(words)
    # p66:前終端記号を追加
    for i, word in enumerate(words):
        for lhs, prob in preterm[word]:
            best_score[f'{i}|{i+1}|{lhs}'] = prob

    # 非終端記号の組み合わせ
    for j in range(2, len(words) + 1):
        for i in range(j-2, -1, -1):   # スパンの左側(右から左へ処理)
            for k in range(i+1, j):
                for sym, lsym, rsym, prob in nonterm:
                    key = f'{i}|{j}|{sym}'
                    l_key = f'{i}|{k}|{lsym}'
                    r_key = f'{k}|{j}|{rsym}'
                    if (l_key in best_score) and (r_key in best_score):
                        new_prob = best_score[l_key] + best_score[r_key] + prob
                        if (key not in best_score) or (new_prob > best_score[key]):
                            best_score[key] = new_prob
                            best_edge[key] = (l_key, r_key)

    return create_s_tree(f'0|{len_wds}|S', best_edge, words)


def create_s_tree(key, best_edge, words):
    sym = key.split('|')[2]
    if key in best_edge:
        lkey, rkey = best_edge[key]
        lstruct = create_s_tree(lkey, best_edge, words)
        rstruct = create_s_tree(rkey, best_edge, words)
        return f'({sym} {lstruct} {rstruct})\t'

    else:
        i = int(key.split('|')[0])
        return f'({sym} {words[i]})\t'



if __name__ == '__main__':
    grammar_file = sys.argv[1]   # {../data/wiki-en-test.grammar, ../test/08-grammar.txt}
    input_file = sys.argv[2]   # {../data/wiki-en-short.tok, ../test/08-input.txt}
    grammar = load_grammar(grammar_file)

    with open(input_file, 'r', encoding='utf-8') as input_f, open('my_ans.txt', 'w', encoding='utf-8') as ans_f:

        for line in input_f.readlines():
            words = line.strip().split()
            print(predict_s_tree(words, grammar), file=ans_f)


