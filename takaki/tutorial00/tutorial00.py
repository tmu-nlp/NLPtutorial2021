import sys
from collections import defaultdict


def solve(path):
    d = defaultdict(lambda: 0)
    with open(path) as f:
        for w in f.read().split():
            d[w] += 1
    return d


def test(path_input, path_answer):
    with open(path_answer) as f:
        answer = {x[0]: int(x[1]) for l in f.readlines() if (x := l.split())}
    result = solve(path_input)
    return answer == result


if __name__ == '__main__':
    # assert test("00-input.txt", "00-answer.txt")
    for k, v in solve(sys.argv[1]).items():
        print(k, v)
