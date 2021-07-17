import sys
from collections import defaultdict
import os

os.chdir(os.path.dirname(os.path.realpath('___file__')))

def message(text):
    print(text)


def cnt_words(path):
    cnter = defaultdict(int)
    total_cnt = 0
    with open(path, 'r', encoding='utf-8') as train_f:
        for line in train_f:
            words = line.rstrip().split() + ['</s>']   # 文末にEOSを追加
            for word in words:
                cnter[word] += 1          # 出現頻度を数える
                total_cnt += 1
    return cnter, total_cnt


def train_unigram(train_path, out_path):
    cnter, total = cnt_words(train_path)
    with open(out_path, 'w', encoding='utf-8') as train_o:
        for k, v in sorted(cnter.items()):
            train_o.write(f'{k}\t{v / total:.3f}\n')


if __name__ == '__main__':
    message('Using wiki file to train unigram model.')
    train_path = '../data/wiki-ja-train.word'
    out_path = './uni_probs.txt'
    train_unigram(train_path, out_path)
    message('unigram training Finished!')







