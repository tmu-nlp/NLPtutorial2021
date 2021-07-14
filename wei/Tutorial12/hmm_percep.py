import math
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm

def arguments_parser():
# use module argparse to write command-line interfaces.
# parser for command-line options, arguments and sub-commands
    parser = argparse.ArgumentParser()
    parser.add_argument('train_file', help='学習用ファイル', type=str)
    parser.add_argument('input_file',help='入力ファイル', type=str)
    return parser.parse_args()

def load_train_data(train_path):
    with open(train_path, 'r', encoding='utf-8') as train_f:
        for line in train_f:
            elements = line.rstrip('\n').split()
            X = []
            Y = []
            for element in elements:     # Natural_JJ
                word, tag = element.split('_')
                X.append(word)
                Y.append(tag)
            yield X, Y

def load_test_data(test_path):
    with open(test_path, 'r', encoding='utf-8') as test_f:
        for line in test_f:
            X = line.strip('\n').split()
            yield X

# 3 key elements in HMM:
# initial state probability vector -> [y0],
# state transition probability matrix -> [p(y_(t+1)=s_j|y_t=s_i)]
# emission probability matrix -> [p(x_t=o_i|y_t=s_j)]
def create_trans(tag1, tag2):
    return [f'T,{tag1},{tag2}']

def create_emit(tag, word):
    result = [f'E,{tag},{word}']
    if word[0].isupper():
        result.append(f'CAPS,{tag}')
    return result

# p29: 素性を使ったビタビアルゴリズムの構築
def hmm_viterbi(w, X, transition, tags):
    l = len(X)      # 単語数
    best_score = {'0 <s>': 0}     # <s>から開始、辞書を作成
    best_edge = {'0 <s>': None}

    for i in range(l):
        for prev in tags:
            for next_ in tags:
                if f'{i} {prev}' not in best_score or f'{prev} {next_}' not in transition:
                    continue
                score = best_score[f'{i} {prev}'] + \
                        sum(w[key] for key in create_trans(prev, next_) + create_emit(next_, X[i]))
                if f'{i+1} {next_}' not in best_score or best_score[f'{i+1} {next_}'] < score:
                    best_score[f'{i+1} {next_}'] = score
                    best_edge[f'{i+1} {next_}'] = f'{i} {prev}'

    # </s>に対して同様の処理
    for tag in tags:
        if f'{tag} </s>' not in transition:
            continue
        score = best_score[f'{l} {tag}'] + \
            sum(w[key] for key in create_trans(tag, '</s>'))
        if f'{l+1} </s>' not in best_score or best_score[f'{l+1} </s>'] < score:
            best_score[f'{l+1} </s>'] = score
            best_edge[f'{l+1} </s>'] = f'{l} {tag}'


    tag_path = []
    next_edge = best_edge[f'{l+1} </s>']
    while next_edge != '0 <s>':
        _, tag = next_edge.split()
        tag_path.append(tag)
        next_edge = best_edge[next_edge]
    tag_path.reverse()

    return tag_path

# P28により、素性辞書を構築
def create_features(X, Y):
    phi = defaultdict(int)
    Y_ = ['<s>'] + Y + ['</s>']
    for i in range(len(Y_)-1):
        first_tag = Y_[i]                    # '<s>'
        next_tag = Y_[i+1]
        for key in create_trans(first_tag, next_tag):
            phi[key] += 1                   # 連続した品詞タグを数える
        if i == len(Y_)-2:                  # 最後の品詞タグまで中断
            break
        for key in create_emit(Y[i], X[i]):
            phi[key] += 1                   # 品詞タグ_単語対を数える
    return phi

# hmm_perceptron モデル学習
def train_hmm_percep(train_path, epoch):
    transition = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for _, tags in load_train_data(train_path):
        tags_ = ['<s>'] + tags + ['</s>']
        for i in range(len(tags_)-1):
            transition[f'{tags_[i]} {tags_[i+1]}'] += 1   # POS transition
        possible_tags.update(tags)

    # p26
    w = defaultdict(int)
    for i in tqdm(range(epoch)):
        for X, Y_prime in load_train_data(train_path):
            Y_hat = hmm_viterbi(w, X, transition, possible_tags)  # prediction tag list
            phi_prime = create_features(X, Y_prime)
            phi_hat = create_features(X, Y_hat)
            for k, v in phi_prime.items():
                w[k] += v
            for k, v in phi_hat.items():
                w[k] -= v
    # trained model
    with open('hmm_percep.model', 'wb') as model_f:
        pickle.dump((dict(transition), possible_tags, w), model_f)


def test_hmm_percep(test_path):
    with open('hmm_percep.model', 'rb') as model_f:
        transition, possible_tags, w = pickle.load(model_f)

    with open('answer.txt', 'w', encoding='utf-8') as out_f:
        for X in load_test_data(test_path):
            Y_hat = hmm_viterbi(w, X, transition, possible_tags)
            print(''.join(Y_hat), file=out_f)

def main():
    args = arguments_parser()
    train_path = args.train_file
    test_path = args.input_file

    train_hmm_percep(train_path,epoch=5)
    test_hmm_percep(test_path)


if __name__ == '__main__':
    main()




