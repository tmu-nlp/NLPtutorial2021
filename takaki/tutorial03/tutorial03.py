import sys
sys.path.append("../")
from tutorial01.tutorial01 import Tutorial01
from math import log2


class Tutorial04:
    unigram = Tutorial01()

    def train(self, path):
        self.unigram.train(path)

    def forward(self, line, λ1=0.95, V=1000000):
        best_edge, best_score = {0: None}, {0: 0}
        for word_end in range(1, len(line) + 1):
            best_score[word_end] = 10 ** 10
            for word_begin in range(word_end):
                word = line[word_begin:word_end]
                if word in self.unigram.model or len(word) == 1:
                    prob = self.unigram.prob(word, λ1, V)
                    score = best_score[word_begin] - log2(prob)
                    if score < best_score[word_end]:
                        best_score[word_end] = score
                        best_edge[word_end] = (word_begin, word_end)
        return best_edge

    def backward(self, line, best_edge):
        words = []
        next_edge = best_edge[len(best_edge) - 1]
        while next_edge != None:
            word = line[next_edge[0]:next_edge[1]]
            words.append(word)
            next_edge = best_edge[next_edge[0]]
        return ' '.join(reversed(words))

    def store(self, path):
        self.unigram.store(path)

    def restore(self, path):
        self.unigram.restore(path)


if __name__ == '__main__':
    x = Tutorial04()
    x.restore('data/04-model.txt')
    with open('data/04-input.txt', 'r') as f:
        lines = f.readlines()
    with open('data/test-answer.txt', 'w') as f:
        f.writelines([
            f'{x.backward(l, x.forward(l))}\n'
            for line in lines
            if (l := line.strip())
        ])

    x = Tutorial04()
    x.train('data/wiki-ja-train.word')
    with open('data/wiki-ja-test.txt', 'r') as f:
        lines = f.readlines()
    with open('data/wiki-answer.txt', 'w') as f:
        f.writelines([f'{x.backward(l, x.forward(l))}\n' for l in lines])
    '''
    Sent Accuracy: 0.00% (/84)
    Word Prec: 46.84% (371/792)
    Word Rec: 57.61% (371/644)
    F-meas: 51.67%
    Bound Accuracy: 65.25% (552/846)
    '''
