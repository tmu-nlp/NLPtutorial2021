import sys
import random
from collections import *

def solve(lines):
    dic = defaultdict(lambda: 0)
    for line in lines:
        line = line.strip()
        for word in line.split(" "):
            dic[word] += 1
    dic = sorted(dic.items(), key = lambda x:x[0])
    return dic

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    #lines = open(sys.argv[1], "r")
    result = solve(lines)
    
    #単語の異なり数
    print("There are [{}] words in data/wiki-en-word.".format(len(result)))
    #数単語の頻度
    for _ in range(5):
        word = result[random.randint(0, len(list(result))-1)]
        print("----> {} : {}".format(word[0], word[1]))