from collections import defaultdict
from math import log2
from itertools import permutations

class BigramLangModel():
    def __init__(self):
        self.word_counts = defaultdict(lambda: 0)
        self.context_counts = defaultdict(lambda: 0) # c(in osaka)/c(in)のc(in)
        self.word_probs = defaultdict(lambda: 0)

    def load_file(self, file_pth):
        with open(file_pth) as file:
            sentences = file.readlines()
        words_with_bos_eos = [["<s>"] + sentence.strip().split() + ["</s>"] for sentence in sentences]
        return words_with_bos_eos
    
    def load_model(self, model_pth):
        with open(model_pth) as model:
            for word_prob in model:
                word, prob = word_prob.split("\t")
                self.word_probs[word] = float(prob)
        return self
    
    def train(self, train_file_pth):
        sentences = self.load_file(file_pth=train_file_pth)
        for sentence in sentences:
            for i in range(1, len(sentence)):
                self.word_counts[f"{sentence[i-1]}_{sentence[i]}"] += 1
                self.context_counts[sentence[i-1]] += 1
                self.word_counts[sentence[i]] += 1
                # 訓練データ中の単語数
                self.context_counts[""] += 1
        for ngram, _ in self.word_counts.items():
            words = ngram.split("_")
            if len(words) < 2:
                words = [""] + words
            self.word_probs[ngram] = self.word_counts[ngram] / self.context_counts[words[0]]
            # p(osaka)=c(osaka)/訓練データ中の単語数
            # p(osaka|osaka)=c(in osaka)/c(in)
        return self

    def save(self, prob_file):
        word_tab_prob = [f"{ngram}\t{prob}" for (ngram, prob) in self.word_probs.items()]
        with open(prob_file, mode='w') as file:
            file.write("\n".join(word_tab_prob))
    
    def test(self, test_file_pth, prob_file_pth, λ_1, λ_2):
        self.load_model(model_pth=prob_file)
        V = 1_000_000
        total_words = 0
        log_likelihood = 0
        test_word_with_bos_eos = self.load_file(test_file_pth)
        for sentence in test_word_with_bos_eos:
            for i in range(1, len(sentence)):
                # unigram
                p_1 = λ_1 * self.word_probs[sentence[i]] + (1 - λ_1) / V
                # bigram
                p_2 = λ_2 * self.word_probs[f"{sentence[i-1]}_{sentence[i]}"] + (1 - λ_2) * p_1
                log_likelihood -= log2(p_2)
                total_words += 1
        entropy = log_likelihood / total_words
        return entropy

if __name__ == "__main__":
    # train
    train_file = "../data/wiki-en-train.word"
    prob_file = "tutorial02.txt"
    model = BigramLangModel()
    model.train(train_file_pth=train_file)
    model.save(prob_file=prob_file)
    
    # test
    test_file = "../data/wiki-en-test.word"
    lamb = [0.05, 0.10 ,0.15 ,0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    entropies = {}
    for lambs in permutations(lamb, 2):
        model = BigramLangModel()
        entropy = model.test(test_file_pth=test_file, prob_file_pth=prob_file, λ_1=lambs[0], λ_2=lambs[1])
        entropies[lambs] = entropy
    for lambs, entropy in sorted(entropies.items(), key=lambda i: i[1]):
        print(lambs, ":", entropy, "\n")
    
'''
(λ_1, λ_2)： entropy

(0.85, 0.4) :  9.66524246414735

(0.8, 0.4) :  9.666736125857843

(0.85, 0.3) :  9.674816615828966

(0.8, 0.3) :  9.678455239875438

(0.9, 0.4) :  9.683859593544033

(0.9, 0.3) :  9.69136459909659

(0.8, 0.5) :  9.702471977919986

(0.85, 0.5) :  9.702613526249054

(0.7, 0.4) :  9.703528449350763

(0.7, 0.3) :  9.71978989692158
'''