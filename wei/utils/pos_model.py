import os, sys
from collections import defaultdict
import pickle, pprint
import math


sys.path.append(os.path.pardir)

def load_data(file_path, mode='train', decorator=None):
    '''
    return [[(word, pos), ...'\n'],[(word, pos), ...'\n'],...]
    '''
    if decorator is None:
        decorator = lambda w: w

    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            pairs = line.strip('\n').split(' ')
            pairs.insert(0, '<s>_BOS')
            pairs.append('</s>_EOS')

            line_data = []
            for pair in pairs:
                if mode == 'train':
                    word, pos = pair.split('_')
                    word = decorator(word)
                    line_data.append((word, pos))
                elif mode == 'test':
                    word = pair.split('_')[0]
                    word = decorator(word)
                    line_data.append(word)
            data.append(line_data)

    return data


def iterate_ngram(seq, n=2):
    for gram in zip(*[seq[i:] for i in range(n)]):
        yield gram


def defaultdict_int():
    return defaultdict(int)


class PosModel:
    def __init__(self):
        self.Pt = None
        self.Pe = None
        self.lamb = None
        self.unk_rate = None

    def train(self, data, vocab_size=1e6, lamb=0.95):
        self.Pt = self.__train_Pt(data)
        self.Pe = self.__train_Pe(data)
        self.lamb = lamb
        self.unk_rate = 1 / vocab_size

    # chapter04-p8:learning algorithm
    def __train_Pt(self, data):
        pt = defaultdict(defaultdict_int)

        for line in data:
            poses = list(map(lambda p:p[1], line))
            for gram in iterate_ngram(poses, 2):
                pt[gram[0]][gram[1]] += 1       # 遷移を数え上げる


        # frequency to probability
        for k, v in pt.items():
            s = sum(v.values())
            for subk, subv in v.items():
                v[subk] = subv / s
        print(pt)
        return pt

    def __train_Pe(self, data):
        pe = defaultdict(defaultdict_int)

        for line in data:
            for pair in line:
                word = pair[0]
                pos = pair[1]

                pe[pos][word] += 1           # 生成を数え上げる

        for k, v in pe.items():
            s = sum(v.values())
            for subk, subv in v.items():
                v[subk] = subv / s

        return pe

    def save_params(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump((self.Pt, self.Pe, self.lamb, self.unk_rate), f)

    def load_params(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
            self.Pt, self.Pe, self.lamb, self.unk_rate = params
            pprint.pprint(params)

    def predict_pos(self, data_):
        yield self.__viterbi(data_)

    def __viterbi(self, data):
        results = []
        for line in data:
            # 前向きステップ
            best_edges = {}
            best_scores = defaultdict(lambda: 10**10)

            best_edges['0 BOS'] = None
            best_scores['0 BOS'] = 0

            prev_poses = line[0][1]
            for i, word in enumerate(line[:], 1):
                next_prev_poses = []
                for prev_pos in prev_poses:
                    prev_node_key = f'{i} {prev_pos}'
                    line_poses = self.Pt[prev_pos].keys()
                    print(f'{i+1} : {prev_pos} => {list(line_poses)}')
                    for line_pos in line_poses:
                        node_key = f'{i+1} {line_pos}'

                        # chapter04-p9:パスの尤度を計算及び平滑化
                        score = best_scores[prev_node_key]
                        score += -math.log2(self.Pt[prev_pos][line_pos])
                        score += -math.log2(self.lamb * self.Pe[line_pos][word] + (1 - self.lamb) * self.unk_rate)

                        # 最短経路を算出
                        if best_scores[node_key] > score:
                            print(f'{i+1} : {best_scores[node_key]} > {score}')
                            best_scores[node_key] = score
                            best_edges[node_key] = prev_node_key

                    next_prev_poses += line_poses
                    prev_poses = list(set(next_prev_poses))

            print(best_edges)
            # 後ろ向きステップ
            tags = []
            next_edge = line[-1]

            while next_edge != None:
                position,tag = next_edge.split(' ')
                tags.append(tag)
                next_edge = best_edges[next_edge]
            tags.reverse()
            results.append(' '.join(tags) + '\n')

        return results


if __name__ == '__main__':
    file_path = '../data/wiki-en-train.norm_pos'
    data = load_data(file_path)[0]


























