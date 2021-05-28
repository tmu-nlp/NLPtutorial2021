from collections import defaultdict
from math import log2


class Tutorial01:
    model = {}

    def train(self, path):
        counts = defaultdict(int)
        total_count = 0
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = line.split() + ['</s>']
            for word in words:
                counts[word] += 1
                total_count  += 1
        for word in counts:
            prob = float(counts[word]) / total_count
            self.model[word] = prob

    def prob(self, word, λ1=0.95, V=1000000):
        P = (1 - λ1) / V
        if word in self.model:
            P += λ1 * self.model[word]
        return P

    def test(self, path, λ1=0.95, V=1000000):
        W, H, U = 0, 0, 0
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = line.split() + ['</s>']
            for word in words:
                W += 1
                P = self.prob(word, λ1, V)
                if word not in self.model:
                    U += 1
                H -= log2(P)
        print(f'entropy : {H / W}')
        print(f'coverage: {(W - U) / W}')

    def store(self, path):
        with open(path, 'w') as f:
            f.writelines([f'{w} {p}\n' for w, p in self.model.items()])

    def restore(self, path):
        with open(path, 'r') as f:
            self.model = {
                x[0]: float(x[1]) for l in f.readlines() if (x := l.split())
            }


if __name__ == '__main__':
    x = Tutorial01()
    x.train('data/01-train-input.txt')
    x.store('data/model.txt')
    x.test('data/01-test-input.txt')
    '''
    entropy : 6.709899494272102
    coverage: 0.8
    '''

    print('----------')

    x = Tutorial01()
    x.train('data/wiki-en-train.word')
    x.store('data/wiki-en-model.txt')
    x.test('data/wiki-en-test.word')
    '''
    entropy : 10.527337238682652
    coverage: 0.895226024503591
    '''
