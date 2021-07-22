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

def update_weights(weights, predict, correct, feats):
    for name, value in feats.items():
        weights[predict][name] -= value
        weights[correct][name] += value
        return 0 


def shift_reduce_train(queue, heads, weights):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        shift, r_left, r_right = calculate_score(feats, weights)
        if len(stack) < 2:
            correct = 'shift'
        elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
            correct = 'right'
        elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
            correct = 'left'
        else:
            correct = 'shift'
        
        if shift >= r_left and shift >= r_right and len(queue) > 0:
            predict = 'shift'
        elif r_left >= r_right:
            predict = 'left'
        else:
            predict = 'right'
        
        if predict != correct:
            update_weights(weights, predict, correct, feats)
        if correct == 'shift':
            stack.append(queue.popleft())
        elif correct == 'left':
            unproc[stack[-1][0]] -= 1
            del stack[-2]
        elif correct == 'right':
            unproc[stack[-2][0]] -= 1
            del stack[-1]

    return weights

def shift_reduce_test():
    return 0

if __name__ == "__main__":
    train = '/Users/michitaka/lab/NLP_tutorial/nlptutorial/data/mstparser-en-train.dep'
    

    weights = {}
    weights['shift'] = defaultdict(int)
    weights['left'] = defaultdict(int)
    weights['right'] = defaultdict(int)
    queue_and_heads = []
    queue = deque()
    heads = [-1]
    with open(train) as train:
        for line in train:
            #line = line.strip().split('\t')
            if line == '\n':
                queue_and_heads.append((queue, heads))
                queue = deque()
                heads = [-1]
            else:
                id, word, _, pos, _, _, parent, _ = line.rstrip('\n').split('\t')
                queue.append((int(id), word, pos))
                heads.append(int(parent))
    for _ in tqdm(range(20)):
        for queue, heads in queue_and_heads:
            weights = shift_reduce_train(queue.copy(), heads, weights)
    with open('/Users/michitaka/lab/NLP_tutorial/tutorial11/weights_file', 'wb') as weights_file:
        dill.dump(weights, weights_file)