from collections import defaultdict
from pprint import pprint


path1 = 'data/wiki-en-test.pos'
path2 = 'data/result.pos'
pathx = 'data/wiki-en-test.norm'


def debug(pos1, pos2):
    with open(path1) as f1, open(path2) as f2, open(pathx) as fx:
        for l1, l2, lx in zip(f1.readlines(), f2.readlines(), fx.readlines()):
            for w1, w2, wx in zip(l1.split(), l2.split(), lx.split()):
                if w1 == pos1 and w2 == pos2:
                    print('\033[31m'+ wx + '\033[0m', end=' ')
                else:
                    print(wx, end=' ')
            print('')


debug('NNP', 'NN')
