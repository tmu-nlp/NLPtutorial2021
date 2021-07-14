from random import randint,random
from collections import defaultdict, Counter
import math



topic_word_cnt = defaultdict(int)
doc_topic_cnt = defaultdict(int)


def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as data:
        for i, line in enumerate(data):
            line = line.rstrip()
            for j, word in enumerate(line.split()):
                yield i, j, word        # doc_id,word_id,word


# p23: function add_counts(word, topic, docid, amount)
def add_counts(wijt, amount):
    (doc_id, word_id, word), topic = wijt
    topic_word_cnt[topic] += amount     # counts topics
    topic_word_cnt[f'{word}|{topic}'] += amount      # counts words in a topic
    doc_topic_cnt[doc_id] += amount
    doc_topic_cnt[f'{topic}|{doc_id}'] += amount

# P14
def sample_one(probs):
    z = sum(probs)              # 確率の和(正規化項)を計算
    r = random()*z              # [0,z)の乱数を一様分布によって生成
    for k in range(len(probs)):
        r -= probs[k]           # 現在の確率を引く
        if r <= 0:
            return k

# p21: Dirichlet Allocationによる平滑化
def get_probs(wijt, num_words, num_topics, alpha, beta):
    probs = []
    (doc_id, word_id, word), _ = wijt
    for topic in range(num_topics):
        cnt_t = topic_word_cnt[topic]
        cnt_t_w = topic_word_cnt[f'{word}|{topic}']
        prob_t_w = (cnt_t_w + alpha) / (cnt_t + alpha * num_words)

        cnt_d = doc_topic_cnt[doc_id]
        cnt_d_t = doc_topic_cnt[f'{topic}|{doc_id}']
        prob_d_t = (cnt_d_t + beta) / (cnt_d + beta * num_topics)

        probs.append(prob_d_t * prob_t_w)
    return probs


def train(data_path, num_topics, num_epochs, alpha, beta):

    w2t = {wij:randint(0, num_topics -1) for wij in load_data(data_path)}
    vocab = Counter(w[-1] for w in w2t.keys())

    for wijt in w2t.items():
        add_counts(wijt, 1)

    num_words = len(vocab)
    for e in range(num_epochs):
        corpus_entropy = 0
        for wijt in w2t.items():
            add_counts(wijt, -1)
            probs = get_probs(wijt, num_words, num_topics,alpha, beta)
            topic = sample_one(probs)
            corpus_entropy += -math.log2(probs[topic])
            w2t[wijt[0]] = topic
            add_counts((wijt[0], topic), 1)
    print(f'Epoch{e} entropy:{corpus_entropy:.2f}')

if __name__ == '__main__':
    train_file = '../data/wiki-en-documents.word'
    test_file = '../test/07-train.txt'
    train(train_file, 2, 10, 1, 1)