from collections import defaultdict
import numpy as np
from math import log2
import string
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

class LDA():
    def __init__(self, n_topics) -> None:
        self.n_topics = n_topics
        self.word_counts = defaultdict(lambda: 0) # {単語_トピックorトピック：コーパスでの出現回数}
        self.topic_counts = defaultdict(lambda: 0) # {ドキュメントIDorトピック_ドキュメントID：出現回数}
        self.word_corpus = [] # [[word, ...], [word, ...]]
        self.topic_corpus = [] # [[topic, ...], [topic, ...]]
        self.α = 0.01
        self.β = 0.01

    def load_file(self, file_pth):
        with open(file_pth, "r", encoding="utf-8") as f:
            sentence = [line.strip().split() for line in f]
            return sentence
    
    def add_counts(self, word, topic, doc_id, amount):
        self.word_counts[f"{topic}"] += amount
        self.word_counts[f"{word}|{topic}"] += amount

        self.topic_counts[f"{doc_id}"] += amount
        self.topic_counts[f"{topic}|{doc_id}"] += amount

    def sample_one(self, probs):
        z = sum(probs)
        remaining = np.random.uniform(0, z)
        for i in range(len(probs)):
            remaining -= probs[i]
            if remaining <= 0:
                return i
    
    def initialize(self, file_pth):
        for words in self.load_file(file_pth):
            doc_id = len(self.word_corpus)
            topics = []
            for word in words:
                topic = np.random.randint(self.n_topics)
                topics.append(topic)
                self.add_counts(word, topic, doc_id, 1)
            self.word_corpus.append(words)
            self.topic_corpus.append(topics)

    def sampling(self, file_pth, iter=50):
        self.initialize(file_pth)
        for i in tqdm(range(iter)):
            ll = 0
            for doc_id in range(len(self.word_corpus)):
                for word_id in range(len(self.topic_corpus[doc_id])):
                    word = self.word_corpus[doc_id][word_id]
                    topic = self.topic_corpus[doc_id][word_id]
                    # 最初はtopicはとりあえず置いているだけ
                    self.add_counts(word, topic, doc_id, -1)
                    probs = []
                    for n_topic in range(self.n_topics):
                        p_word = (self.word_counts[f"{word}|{n_topic}"] + self.α) / (self.word_counts[f"{n_topic}"] + self.α*len(self.word_counts))
                        p_topic = (self.topic_counts[f"{n_topic}|{doc_id}"] + self.β) / (self.topic_counts[f"{doc_id}"] + self.β*len(self.topic_counts))
                        p_xy = p_word * p_topic
                        probs.append(p_xy)
                    new_y = self.sample_one(probs)
                    ll += log2(probs[new_y])
                    self.add_counts(word, new_y, doc_id, 1)
                    self.topic_corpus[doc_id][word_id] = new_y
            print(ll)
    
    def answer(self, file_pth, categolized_file_pth):
        topics = [[] for _ in range(self.n_topics)]
        with open(file_pth, 'w') as f:
            for doc_id in range(len(self.word_corpus)):
                for word_id in range(len(self.topic_corpus[doc_id])):
                    word = self.word_corpus[doc_id][word_id]
                    topic = self.topic_corpus[doc_id][word_id]
                    f.write(f"{word}_{topic}"+"\n")
                    topics[topic].append(f"{word}")
        
        stop_words = set(stopwords.words("english"))
        punctuations = set(string.punctuation)
        other_punc = set(["&apos;", "&quot;", "&apos;s"])
        stop_words = stop_words.union(punctuations).union(other_punc)
        with open(categolized_file_pth, "w") as f:
            for topic, words in enumerate(topics):
                word_count = {}
                f.write("TOPIC: "+str(topic)+"\n")
                for word in set(words):
                    if word not in stop_words:
                        count = self.word_counts[f"{word}|{topic}"]
                        word_count[word] = count
                word_count = sorted(word_count.items(), key=lambda x:x[1], reverse=True)
                f.write(str(word_count[:20])+"\n\n")                    
                    
if __name__ == "__main__":
    train_file = "data/wiki-­en-documents.word"
    # input_file = "data/07-train.txt"

    model = LDA(8)
    train_file = train_file.replace("\xad", "")
    model.sampling(file_pth=train_file)

    ans_file = "tutorial09/tutorial09.txt"
    categolized_file = "tutorial09/tutorial09_categorized.txt"
    model.answer(file_pth=ans_file, categolized_file_pth=categolized_file)