from collections import defaultdict
from tqdm import tqdm
import pickle

#構造化パーセプトロンアルゴリズム
def train_hmm_percep(words_tags, w, possible_tags, transition):
    # 1文ずつ
    for (words, tags_prime) in words_tags:
        # 今の重みで試しにP_hatを計算して
        tags_hat = hmm_viterbi(w, words, possible_tags, transition)
        # 正しい素性の計算
        phi_prime = create_features(words, tags_prime)
        # 試しに計算してみたpos_hatを使って素性の計算
        phi_hat = create_features(words, tags_hat)
        # 素性の差から重みを更新
        for feat in phi_prime:
            w[feat] += (phi_prime[feat] - phi_hat[feat])
    return

#素性を使ったビタビアルゴリズム
def hmm_viterbi(w, words, possible_tags, trans):
    #単語数
    l = len(words)
    #<s>から開始
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

    #同じく、</s>も
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

#構造化パーセプトロンの素性計算
#単語列の素性を構築するCREATURE_FEATURES関数
def create_features(W, P):
    phi = defaultdict(lambda: 0)
    for i, pos in enumerate(range(len(P)+1)):
        if i == 0:
            first_tag = '<s>'
        else:
            first_tag = P[i-1]
        if i == len(P):
            next_tag = '</s>'
        else:
            next_tag = P[i]
        phi[f'T {first_tag} {next_tag}'] += 1

    for i in range(len(P)):
        phi[f'E {P[i]} {W[i]}']
        if start_caps(W[i]):
            phi[f'CAPS {P[i]}'] += 1
    return phi

#back
def start_caps(word):
    if word[0].isupper():
        return True
    else:
        return False


def import_traindata(path):
    '''
    入力: train dataのパス
    出力: [(word1, pos1), (word2, pos2), ...]
    '''
    words_tags = []
    for line in open(path, 'r'):
        word_pos_list = line.rstrip().split()
        words, tags = [], []
        for word_pos in word_pos_list:
            word, pos = word_pos.split('_')
            words.append(word)
            tags.append(pos)
        words_tags.append((words, tags))

    return words_tags

#hmm_perceptronモデルを学習
def preprocess(words_tags):
    trans = defaultdict(int)
    possible_tags = {'<s>', '</s>'}
    for _, tags in words_tags:
        prevs, nexts = ['<s>'] + tags, tags + ['</s>']
        for prv, nxt in zip(prevs, nexts):
            trans[f'{prv} {nxt}'] += 1
        possible_tags.update(tags)

    return trans, possible_tags

def import_testdata(path):
    wordslist = []
    for line in open(path, 'r'):
        words = line.rstrip().split()
        wordslist.append(words)
    return wordslist

if __name__ == '__main__':
    trainpath = '/users/kcnco/github/NLPtutorial2021/pan/tutorial12/wiki-en-train.norm_pos'
    testpath = '/users/kcnco/github/NLPtutorial2021/pan/tutorial12/wiki-en-test.norm'
    
    words_tags= import_traindata(trainpath)
    wordslist = import_testdata(testpath)
    trans, possible_tags = preprocess(words_tags)

    epoch = 20

    # 重みの初期化
    w = defaultdict(lambda: 0)

    for _ in tqdm(range(epoch)):
        train_hmm_percep(words_tags, w, possible_tags, trans)

    with open('./weight.txt', 'w+') as f_out:
        for feat, weight in w.items():
            print(f'{feat} {weight}', file=f_out)

    with open('./result.txt', 'w+') as f_out:
        for words in wordslist:
            tags = hmm_viterbi(w, words, possible_tags, trans)
            print(' '.join(tags), file=f_out)

            
            
# Accuracy: 88.76% (4050/4563)
# Most common mistakes:
# NNS --> NN      33
# NN --> JJ       31
# JJ --> VBN      31
# NN --> NNS      24
# NN --> NNP      18
# VBN --> NNS     15
# NN --> VBN      13
# JJ --> NN       13
# JJ --> RB       12
# NNS --> JJ      12
