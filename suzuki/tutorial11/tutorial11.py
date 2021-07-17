from collections import defaultdict
from tqdm import tqdm
import random

def make_feats(stack, queue):
    feats = defaultdict(lambda: 0)
    if len(stack) > 0 and len(queue) > 0:
        w_0 = queue[0][1]
        w_1 = stack[-1][1]
        p_0 = queue[0][2]
        p_1 = stack[-1][2]
        p2_0 = queue[0][3] ##
        p2_1 = stack[-1][3] ##
        feats['W-1{}, W0{}'.format(w_1, w_0)] += 1
        feats['W-1{}, P0{}'.format(w_1, p_0)] += 1
        feats['P-1{}, W0{}'.format(p_1, w_0)] += 1
        feats['P-1{}, P0{}'.format(p_1, p_0)] += 1
        feats['W-1{}, P20{}'.format(w_1, p2_0)] += 1 ##
        feats['P-1{}, P20{}'.format(p_1, p2_0)] += 1 ##
        feats['P2-1{}, W0{}'.format(p2_1, w_0)] += 1 ##
        feats['P2-1{}, P0{}'.format(p2_1, p_0)] += 1 ##
        feats['P2-1{}, P20{}'.format(p2_1, p2_0)] += 1 ##
    if len(stack) > 1:
        w_1 = stack[-1][1]
        w_2 = stack[-2][1]
        p_1 = stack[-1][2]
        p_2 = stack[-2][2]
        p2_1 = stack[-1][3] ##
        p2_2 = stack[-1][3] ##
        feats['W-2{}, W-1{}'.format(w_2, w_1)] += 1
        feats['W-2{}, P-1{}'.format(w_2, p_1)] += 1
        feats['P-2{}, W-1{}'.format(p_2, w_1)] += 1
        feats['P-2{}, P-1{}'.format(p_2, p_1)] += 1
        feats['W-2{}, P2-1{}'.format(w_2, p2_1)] += 1 ##
        feats['P-2{}, P2-1{}'.format(p_2, p2_1)] += 1 ##
        feats['P2-2{}, W-1{}'.format(p2_2, w_1)] += 1 ##
        feats['P2-2{}, P-1{}'.format(p2_2, p_1)] += 1 ##
        feats['P2-2{}, P2-1{}'.format(p2_2, p2_1)] += 1 ##
    return feats

def caluclate_ans(feats, w, queue):
    s_s = 0
    s_l = 0
    s_r = 0
    for name, value in feats.items():
        s_s += w['shift'][name] * value
        s_l += w['left'][name] * value
        s_r += w['right'][name] * value
    if s_s >=  s_l and s_s >= s_r and len(queue) > 0:
        ans = 'shift'
    elif s_l >= s_r:
        ans = 'left'
    else:
        ans = 'right'
    return ans

def caluclate_corr(stack, heads, unproc):
    if len(stack) < 2:
        corr = 'shift'
        return corr
    if heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
        corr = 'right'
    elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
        corr = 'left'
    else:
        corr = 'shift'
    return corr
    
def shift_reduce_train(queue, weights, heads):
    stack = [(0, 'ROOT', 'ROOT', 'ROOT')] ##
    unproc = []
    for i in range(len(heads)): #単語の位置(配列の要素)に単語の子の数を保存
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        ans = caluclate_ans(feats, weights, queue)
        corr = caluclate_corr(stack, heads, unproc)
        if ans != corr:
            for name, value in feats.items():
                weights[ans][name] -= value
                weights[corr][name] += value
        # perform action according to corr
        if corr == 'shift':
            stack.append(queue.pop(0))
        elif corr == 'left':
            unproc[stack[-1][0]] -= 1
            del stack[-2]
        elif corr == 'right':
            unproc[stack[-2][0]] -= 1
            del stack[-1]

def shift_reduce_test(queue, weghts):
    heads = [-1] * (len(queue) + 1)
    stack = [(0, 'ROOT', 'ROOT', 'ROOT')] ##
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        ans = caluclate_ans(feats, weights, queue)
        if len(stack) < 2 or ans == 'shift':
            stack.append(queue.pop(0))
        elif ans == 'left':
            heads[stack[-2][0]] = stack[-1][0]
            del stack[-2]
        elif ans == 'right':
            heads[stack[-1][0]] = stack[-2][0]
            del stack[-1]
    return heads

if __name__ == '__main__':
    data =[] # data = [(queue1, heads1), (queue2, heads2), ...]
    queue = []
    heads = [-1]
    weights = {}
    weights['shift'] = defaultdict(lambda: 0)
    weights['left'] = defaultdict(lambda: 0)
    weights['right'] = defaultdict(lambda: 0)

    iteration = 100

    # make data for training
    with open('mstparser-en-train.dep', 'r') as trainfile:
        for line in trainfile:
            if line != '\n':
                id, surface, base, pos1, pos2, _, parent, label = line.strip().split('\t')
                queue.append((int(id), surface, pos1, pos2)) ##
                heads.append(int(parent))
            else:
                data.append((queue, heads))
                queue = []
                heads = [-1]
    
    # training
    for _ in tqdm(range(iteration)):
        for queue, heads in data:
            shift_reduce_train(queue, weights, heads)

    # make data for test
    data = [] # data = [queue1, queue2, ...]
    queue = []
    ans_data = []
    ans_file_element = []
    with open('mstparser-en-test.dep', 'r') as testfile:
        for line in testfile:
            if line != '\n':
                id, surface, base, pos1, pos2, _, parent, label = line.strip().split('\t')
                queue.append((int(id), surface, pos1, pos2)) ##
                ans_file_element.append((id, surface, base, pos1, pos2, _, parent, label))
            else:
                data.append(queue)
                queue = []
                ans_data.append(ans_file_element)
                ans_file_element = []
    
    # test
    heads_list = []
    for queue in data:
        heads_list.append(shift_reduce_test(queue, weights))
    with open('ans2_file', 'w') as ansfile:
        for i, heads in enumerate(heads_list):
            for j, head in enumerate(heads):
                if j == 0: continue
                id, surface, base, pos1, pos2, _, parent, label = ans_data[i][j-1]
                ansfile.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(id, surface, base, pos1, pos2, _, head, label))
            ansfile.write('\n')

'''

python grade-dep.py mstparser-en-test.dep ans_file

iteration: 100
weight lambda: 0
61.521880% (2854/4639)

iteration: 100
weight lambda: random.randint(0,3)
57.727959% (2678/4639)

素性の拡張
iteration: 100
weight lambda: 0
60.918301% (2826/4639)

itration: 100
weight lambda: random.randint(0,3)
62.211684% (2886/4639)

'''