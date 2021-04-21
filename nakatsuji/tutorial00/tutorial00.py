import sys
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
    ans = solve(lines)
    #for (k, v) in ans:
    #    print(k, v)
    
    #単語の異なり数
    print("Thre are {} words in data/wiki-en-word.".format(len(list(ans))))
    #数単語の頻度
    for (k, v) in ans[1000:1010]:
        print(k, v)