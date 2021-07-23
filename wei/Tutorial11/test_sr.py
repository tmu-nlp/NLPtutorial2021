from collections import defaultdict
from collections import deque
import dill

def ShiftReduce(queue, weights):
    heads = [-1] * (len(queue) + 1)
    stack = [(0, 'ROOT', 'ROOT')]
    while len(queue) > 0 or len(stack) > 1:
        feats = Make_feats(stack, queue)
        s_r, s_l, s_s = calc_score(feats, weights)
        if len(stack) < 2 or (s_s >= s_l and s_s >= s_r and len(queue) > 0):
            stack.append(queue.popleft())
        elif s_l >= s_r:
            heads[stack[-2][0]] = stack[-1][0]
            del stack[-2]
        else:
            heads[stack[-1][0]] = stack[-2][0]
            del stack[-1]
    return heads

def Make_feats(stack, queue):
    feats = defaultdict(lambda: 0)
    if len(stack) > 0 and len(queue) > 0:
        w_0 = queue[0][1]
        w_1 = stack[-1][1]
        p_0 = queue[0][2]
        p_1 = stack[-1][2]
        feats[f'W-1{w_1},W-0{w_0}'] += 1
        feats[f'W-1{w_1},P-0{p_0}'] += 1
        feats[f'P-1{p_1},W-0{w_0}'] += 1
        feats[f'P-1{p_1},P-0{p_0}'] += 1
    if len(stack) > 1:
        w_1 = stack[-1][1]
        w_2 = stack[-2][1]
        p_1 = stack[-1][2]
        p_2 = stack[-2][2]
        feats[f'W-2{w_2},W-1{w_1}'] += 1
        feats[f'W-2{w_2},P-1{p_1}'] += 1
        feats[f'P-2{p_2},W-1{w_1}'] += 1
        feats[f'P-2{p_2},P-1{p_1}'] += 1
    return feats

def calc_score(feats, w):
    s_r = 0
    s_l = 0
    s_s = 0
    for name, value in feats.items():
        s_r += w['right'][name] * value
        s_l += w['left'][name] * value
        s_s += w['shift'][name] * value
    return s_r, s_l, s_s

if __name__ == '__main__':
    with open('model_weights', 'rb') as weights_file:
        weights = dill.load(weights_file)
    data = []
    queue = deque()
    conll = []
    conll_list = []
    with open('../data/mstparser-en-test.dep') as test_file:
        for line in test_file:
            if line == '\n':
                data.append(queue)
                conll_list.append(conll)
                queue = deque()
                conll = []
            else:
                id, word, ori, pos1, pos2, ext, parent, label = line.strip('\n').split('\t')
                queue.append((int(id), word, pos1))
                conll.append([id, word, ori, pos1, pos2, ext, parent, label])
    heads_list = []
    for queue in data:
        heads_list.append(ShiftReduce(queue, weights))
    with open('ans_file', 'w') as ans_file:
        for conll, heads in zip(conll_list, heads_list):
            for types, head in zip(conll, heads[1:]):
                types[6] = str(head)
                line = '\t'.join(types) + '\n'
                ans_file.write(f'{line}')
            ans_file.write('\n')

