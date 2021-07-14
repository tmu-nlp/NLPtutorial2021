from collections import defaultdict
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords

class LDA:
    def __init__(self):
        self.xcorpus = []
        self.ycorpus = []
        self.xcounts = defaultdict(lambda: 0)
        self.ycounts = defaultdict(lambda: 0)
        self.NUM_TYPE = set() #単語タイプ数
        self.NUM_TOPICS = 5 #トピック数

    def initialize(self, train_file):
        with open(train_file) as f:
            for line in f:
                doc_id = len(self.xcorpus)
                words = line.split()
                topics = []
                for word in words:
                    topic = np.random.randint(0, self.NUM_TOPICS) #整数ランダム
                    topics.append(topic)
                    self.add_counts(word, topic, doc_id, amount=1)
                    self.NUM_TYPE.add(word) 
                self.xcorpus.append(words)
                self.ycorpus.append(topics)

    def add_counts(self, word, topic, doc_id, amount):
        self.xcounts[f'{topic}'] += amount
        self.xcounts[f'{word}|{topic}'] += amount
        self.ycounts[f'{doc_id}'] += amount
        self.ycounts[f'{topic}|{doc_id}'] += amount

    def sample_one(self, probs):
        z = sum(probs)
        remaining = np.random.rand() * z
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i

    def sampling(self, iter, α=0.01, β=0.01):
        for _ in tqdm(range(1, iter+1)):
            log_likelihood = 0
            for i in range(len(self.xcorpus)):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, amount=-1) #各カウントの減算
                    probs = [] #各トピックの確率
                    for k in range(self.NUM_TOPICS):
                        Px_k = (self.xcounts[f'{x}|{k}'] + α) / (self.xcounts[f'{k}'] + α*len(self.NUM_TYPE))
                        Pk_y = (self.ycounts[f'{k}|{y}'] + β) / (self.ycounts[f'{y}'] + β*self.NUM_TOPICS)
                        probs.append(Px_k * Pk_y)
                    new_y = self.sample_one(probs)
                    log_likelihood += np.log(probs[new_y])
                    self.add_counts(x, new_y, i, amount=1) #各カウントの加算
                    self.ycorpus[i][j] = new_y

if __name__ == '__main__':
    lda = LDA()
    train_file = '../data/wiki-en-documents.word'
    lda.initialize(train_file)
    lda.sampling(iter=30)

    topic_0 = defaultdict(lambda: 0)
    topic_1 = defaultdict(lambda: 0)
    topic_2 = defaultdict(lambda: 0)
    topic_3 = defaultdict(lambda: 0)
    topic_4 = defaultdict(lambda: 0)
    topics = [topic_0, topic_1, topic_2, topic_3, topic_4]

    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    for i in range(len(lda.xcorpus)):
        for j in range(len(lda.xcorpus[i])):
            if lda.xcorpus[i][j] in stop_words:
                continue
            topic = lda.ycorpus[i][j]
            topics[topic][lda.xcorpus[i][j]] += 1

    with open('./res_topic.txt', 'w') as f:
        for i, topic in enumerate(topics):
            f.write(f'topic{i}\n')
            f.write(f'{sorted(topic.items(), key=lambda x:x[1], reverse=True)}\n')