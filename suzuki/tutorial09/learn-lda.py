import random
import math
from collections import defaultdict
from tqdm import tqdm
import string

xcorpus = [] # 文ごとの単語リストのリスト
ycorpus = [] # 文ごとのトピックリストのリスト
xcounts = defaultdict(lambda: 0) # 単語 と 単語|トピック の頻度
ycounts = defaultdict(lambda: 0) # 文書id と トピック|文書id の頻度

def sample_one(probs):
    z = sum(probs)
    remaining = random.random() * z
    for i in range(len(probs)):
        remaining -= probs[i]
        if remaining <= 0:
            return i
    raise Exception('Error at sample_one')

def add_counts(word, topic, docid, amount):
    xcounts[topic] += amount
    xcounts['{}|{}'.format(word, topic)] += amount
    ycounts[docid] += amount
    ycounts['{}|{}'.format(topic, docid)] += amount
    if xcounts[topic] < 0 or ycounts[topic] < 0 or xcounts['{}/{}'.format(word, topic)] < 0 or ycounts['{}/{}'.format(topic, docid)] < 0:
        raise Exception('Error at add_counts')

def prob_of_topic(x, k, Y, alpha, beta):
    Nx = len(xcorpus)
    Ny = len(ycorpus)
    p_x_given_k = (xcounts['{}|{}'.format(x, k)] + alpha) / (xcounts[k] + alpha * Nx)
    p_y_given_Y = (ycounts['{}|{}'.format(y, Y)] + beta) / (ycounts[Y] + beta * Ny)
    return p_x_given_k * p_y_given_Y


if __name__ == '__main__':
    input_file = 'wiki-en-documents.word'
    num_topics = 5
    iterations = 30
    alpha = 0.02
    beta = 0.02
    table = str.maketrans("", "", string.punctuation)

    # 初期化
    with open(input_file, 'r') as file:
        for line in file:
            docid = len(xcorpus) # document id
            line = line.strip()
            line = line.translate(table)
            words = line.split()

            topics = []
            for word in words:
                topic = random.randint(0, num_topics - 1)
                topics.append(topic)
                add_counts(word, topic, docid, 1)
            xcorpus.append(words)
            ycorpus.append(topics)

    #サンプリング
    for _ in tqdm(range(iterations)):
        ll = 0
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                add_counts(x, y, i, -1) #各カウントの減算
                probs = []
                for k in range(num_topics):
                    probs.append(prob_of_topic(x, k, i, alpha, beta))
                new_y = sample_one(probs)
                ll += math.log(probs[new_y])
                add_counts(x, new_y, i, 1)
                ycorpus[i][j] = new_y
        print(ll)
    
    with open('09-answer', 'w') as ans:
        for i in range(len(xcorpus)):
            for j in range(len(xcorpus[i])):
                x = xcorpus[i][j]
                y = ycorpus[i][j]
                print('{}_{}'.format(x, y), end = ' ', file = ans)
            print(file = ans)