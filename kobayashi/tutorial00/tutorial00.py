import sys
from collections import defaultdict

file = open(sys.argv[1])
dic = defaultdict(lambda: 0)

for line in file:
    for word in line.strip().split():
        dic[word] += 1

print("\n".join([f"{w} {c}" for w, c in sorted(dic.items())]))