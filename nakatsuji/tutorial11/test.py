from collections import *
from tqdm import tqdm
import dill
def make_feats(stack, queue):
    feats = defaultdict(int)
    if len(stack) > 0 and len(queue) > 0:
        w_0, p_0 = queue[0][1], queue[0][2]
        w_1,p_1 = stack[-1][1], stack[-1][2]
        feats[f'W-1{w_1},W-0{w_0}'] += 1
        feats[f'W-1{w_1},P-0{p_0}'] += 1
        feats[f'P-1{p_1},W-0{w_0}'] += 1
        feats[f'P-1{p_1},P-0{p_0}'] += 1
    if len(stack) > 1:
        w_1, p_1 = stack[-1][1], stack[-1][2]
        w_2, p_2 = stack[-2][1], stack[-2][2]
        feats[f'W-2{w_2},W-1{w_1}'] += 1
        feats[f'W-2{w_2},P-1{p_1}'] += 1
        feats[f'P-2{p_2},W-1{w_1}'] += 1
        feats[f'P-2{p_2},P-1{p_1}'] += 1
    return feats

def calculate_score(feats, w):
    shift, r_left, r_right = 0, 0, 0
    for k, v in feats.items():
        shift += w['shift'][k] * v
        r_left += w['left'][k] * v
        r_right += w['right'][k] * v
    return shift, r_left, r_right

def shit_reduce_test(queue, weights):
    heads = [-1] * (len(queue) + 1)
    stack = [(0, 'ROOT', 'ROOT')]
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        shift, r_left, r_right = calculate_score(feats, weights)
        if len(stack) < 2 or (shift >= r_left and shift >= r_right and len(queue) > 0):
            stack.append(queue.popleft())
        elif r_left >= r_right:
            heads[int(stack[-2][0])] = stack[-1][0]
            del stack[-2]
        else:
            heads[int(stack[-1][0])] = stack[-2][0]
            del stack[-1]
    return heads

if __name__ == '__main__':
    test = '/Users/michitaka/lab/NLP_tutorial/nlptutorial/data/mstparser-en-test.dep'

    with open('/Users/michitaka/lab/NLP_tutorial/tutorial11/weights_file', 'rb') as weights_file:
        weights = dill.load(weights_file)
    queues = []
    queue = deque()
    tests = []
    a_test = []
    with open(test) as test_file, open('/Users/michitaka/lab/NLP_tutorial/tutorial11/ans.txt', 'w', encoding="utf-8") as f:
        for line in test_file:
            if line == '\n':
                queues.append(queue)
                queue = deque()
            else:
                id, word, _, pos, _, _, parent, _ = line.rstrip('\n').split('\t')
                queue.append((id, word, pos))
        for queue in queues:
            heads = shit_reduce_test(queue, weights)
            heads.pop(0)
            for i in range(len(heads)):
                print(f'{i}\t_\t_\t_\t_\t_\t{heads[i]}\t_\n', file=f)
            print('\n', file=f)


    