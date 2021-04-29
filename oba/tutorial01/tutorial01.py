import sys
from collections import defaultdict

class UnigramLangModel():
    def __init__(self):
        self.word_counts = defaultdict(lambda: 0)
        self.total_words = 0
        self.word_probs = defaultdict(lambda: 0)

    def load_file(self, file_pth):
        with open(file_pth) as file:
            sentences = file.readlines()
        #appendはリストを一度変数に格納してからでないと使えない
        words_with_eos = [sentence.strip().split() + ["</s>"] for sentence in sentences]
        return words_with_eos

    def train(self, train_file_pth):
        sentences = self.load_file(file_pth=train_file_pth)
        for sentence in sentences:
            for word in sentence:
                self.word_counts[word] += 1
                self.total_words += 1
        print("trained")


if __name__ == "__main__":
    # train_file = sys.argv[1]
    train_file = "../data/wiki-en-train.word"

    model = UnigramLangModel()
    model.train(train_file_pth=train_file)