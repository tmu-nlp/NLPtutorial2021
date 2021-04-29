import sys
from collections import defaultdict
from math import log2

class UnigramLangModel():
    def __init__(self):
        self.word_counts = defaultdict(lambda: 0)
        self.total_words = 0
        self.word_probs = defaultdict(lambda: 0)
        self.λ_1 = 0.95
        self.λ_unk = 1 - self.λ_1

    def load_file(self, file_pth):
        with open(file_pth) as file:
            sentences = file.readlines()
        #appendはリストを一度変数に格納してからでないと使えない
        words_with_eos = [sentence.strip().split() + ["</s>"] for sentence in sentences]
        return words_with_eos

    def prob(self):
        for word, num_of_word in self.word_counts.items():
            self.word_probs[word] = num_of_word / self.total_words
        print("probed")
        return self

    def train(self, train_file_pth):
        sentences = self.load_file(file_pth=train_file_pth)
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
                self.total_words += 1
        print("trained")

    def save(self, prob_file):
            self.prob()
            word_tab_prob = [f"{word}\t{prob}" for (word, prob) in self.word_probs.items()]
            with open(prob_file, mode='w') as file:
                file.write("\n".join(word_tab_prob))
            print("saved")
    
    def load_model(self, model_pth):
        with open(model_pth) as model:
            for word_prob in model:
                word, prob = word_prob.split("\t")
                self.word_probs[word] = float(prob)

    def test(self, test_file_pth, prob_file_pth):
        self.load_model(model_pth=prob_file)
        num_of_words_with_unk = 1_000_000
        # testファイルに含まれる単語数
        total_words = 0
        # testファイルに含まれるがprobに含まれていない単語数（未知語）
        num_of_unk = 0
        log_likelihood = 0
        test_word_with_eos = self.load_file(test_file_pth)
        for sentence in test_word_with_eos:
            for word in sentence:
                total_words += 1
                prob = self.λ_unk / num_of_words_with_unk
                if word in self.word_probs:
                    prob += self.λ_1 * self.word_probs[word]
                else:
                    num_of_unk += 1
                log_likelihood -= log2(prob)
        entropy = log_likelihood / total_words
        coverage = (total_words - num_of_unk) / total_words
        return(entropy, coverage)

if __name__ == "__main__":
    # train_file = sys.argv[1]
    train_file = "../data/wiki-en-train.word"
    prob_file = "tutorial01.txt"
    test_file = "../data/wiki-en-test.word"

    model = UnigramLangModel()
    model.train(train_file_pth=train_file)
    model.save(prob_file=prob_file)
    
    entropy, coverage = model.test(test_file_pth=test_file, prob_file_pth=prob_file)
    print(f"Entropy: {entropy:.6f}")
    print(f"Coverage: {coverage:.6f}")