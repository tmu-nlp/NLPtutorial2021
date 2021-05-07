from collections import defaultdict
from math import log2

class BigramLangModel():
    def __init__(self):
        self.word_counts = defaultdict(lambda: 0)
        self.context_counts = defaultdict(lambda: 0)
        self.word_probs = defaultdict(lambda: 0)
        self.λ_1 = 0.95
        self.λ_2 = 1 - self.λ_1

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
        print(self.word_probs)
        return self
    
    def train(self, train_file_pth):
        sentences = self.load_file(file_pth=train_file_pth)
        for sentence in sentences:
            for i in range(1, len(sentence)):
                # I_am  
                self.word_counts[f"{sentence[i-1]}_{sentence[i]}"] += 1
                # I
                self.context_counts[sentence[i-1]] += 1
                # am
                self.word_counts[sentence[i]] += 1
                self.context_counts[""] += 1
        for ngram, count in self.word_counts.items():
            words = ngram.split("_")
            if len(words) < 2:
                # am単体
                words = [""] + words
            # I_amの数 / Iの数 
            self.word_probs[ngram] = self.word_counts[ngram] / self.context_counts[words[0]]
        return self

    def save(self, prob_file):
        word_tab_prob = [f"{ngram}\t{prob}" for (ngram, prob) in self.word_probs.items()]
        with open(prob_file, mode='w') as file:
            file.write("\n".join(word_tab_prob))
    
    def test(self, test_file_pth, prob_file_pth):
        self.load_model(model_pth=prob_file)
        V = 1_000_000
        total_words = 0
        log_likelihood = 0
        test_word_with_bos_eos = self.load_file(test_file_pth)
        for sentence in test_word_with_bos_eos:
            for i in range(1, len(sentence)):
                p_1 = self.λ_1 * self.word_probs[sentence[i]]  + (1 - self.λ_1) / V
                p_2 = self.λ_2 * self.word_probs[f"{sentence[i-1]}_{sentence[i]}"] + (1 - self.λ_2) * p_1
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
    model = BigramLangModel()
    entropy = model.test(test_file_pth=test_file, prob_file_pth=prob_file)
    print(entropy)