from collections import defaultdict, deque
import copy
import dill
from tqdm import tqdm

# p13:キューとスタックに基づいて、素性((w_1,w_0),(w_1,p_0),...)を計算
def make_feats(stack, queue):
    feats = defaultdict(lambda :0)
    if len(stack) > 0 and len(queue) > 0:
        w_0 = queue[0][1]      # 未処理の最初
        w_1 = stack[-1][1]     # 処理中の右→最後
        p_0 = queue[0][2]
        p_1 = stack[-1][2]
        feats[f'W-1{w_1}, W-0{w_0}'] += 1  # a girl
        feats[f'W-1{w_1}, P-0{p_0}'] += 1  # a NN
        feats[f'P-1{p_1}, W-0{w_0}'] += 1  # DET girl
        feats[f'P-1{p_1}, P-0{p_0}'] += 1  # DET NN
    if len(stack) > 1:
        w_1 = stack[-1][1]
        w_2 = stack[-2][1]      # 処理中の左→最後から2番
        p_1 = stack[-1][2]
        p_2 = stack[-2][2]
        feats[f'W-2{w_2}, W-1{w_1}'] += 1   # saw a
        feats[f'W-2{w_2}, P-1{p_1}'] += 1   # saw DET
        feats[f'P-2{p_2}, W-1{w_1}'] += 1   # VBD a
        feats[f'P-2{p_2}, P-1{p_1}'] += 1   # VBD DET
    return feats


# 素性関数と重みを掛け合わせて、スコアを計算
def calc_score(feats, w):
    s_r = 0
    s_l = 0
    s_s = 0
    for name, value in feats.items():
        s_r += w['right'][name] * value
        s_l += w['left'][name] * value
        s_s += w['shift'][name] * value
    return s_r, s_l, s_s


# p18:「正解」の計算
def calc_correct(stack, heads, unproc):
    if len(stack) < 2:   # 1単語をキューからスタックへ移動
        correct = 'shift'
    # 左が右の親、且つ右に未処理の子がない
    elif heads[stack[-1][0]] == stack[-2][0] and unproc[stack[-1][0]] == 0:
        correct = 'right'
    # 右が左の親、且つ左に未処理の子がない
    elif heads[stack[-2][0]] == stack[-1][0] and unproc[stack[-2][0]] == 0:
        correct = 'left'
    else:
        correct = 'shift'
    return correct


def calc_predict(s_r, s_l, s_s, queue):
    if s_s >= s_r and s_s >= s_l and len(queue) > 0:
        predict = 'shift'
    elif s_r >= s_l:
        predict = 'right'        # reduce右を実行:左(スタックの1単語目)が右(2単語目)の親-> saw girl
    else:
        predict = 'left'         # reduce左を実行:右(スタックの2単語目)が左(1単語目)の親-> I saw; a girl
    return predict

# 重み更新
def update_weights(w, predict, correct, feats):
    for name, value in feats.items():
        w[predict][name] -= value
        w[correct][name] += value


# p19:shift-reduceの学習アルゴリズム
'''
p14:input of SR algorithm
weights w_s, w_r, w_l
queue = [(1, word1, pos1), (2, word2, pos2),...]
stack = [(0, 'ROOT', 'ROOT')] -> 特別な'ROOT'のみを格納
heads = [-1, head1, head2,...] -> 戻り値は各単語の親のIDを格納したARRAY
'''
def ShiftReduceTrain(queue, heads, weights):
    stack = [(0, 'ROOT', 'ROOT')]
    unproc = []
    for i in range(len(heads)):
        unproc.append(heads.count(i))
    while len(queue) > 0 or len(stack) > 1:
        feats = make_feats(stack, queue)
        s_r, s_l, s_s = calc_score(feats, weights)
        correct = calc_correct(stack, heads, unproc)
        predict = calc_predict(s_r, s_l, s_s, queue)
        if predict != correct:
            update_weights(weights, predict, correct, feats)
        if correct == 'shift':                     # shiftを実行
            stack.append(queue.popleft())
        elif correct == 'left':                    # reduce左を実行
            unproc[stack[-1][0]] -= 1
            del stack[-2]
        elif correct == 'right':                   # reduce右を実行
            unproc[stack[-2][0]] -= 1
            del stack[-1]


if __name__ == '__main__':
    epoch = 50
    weights = {}
    weights['right'] = defaultdict(lambda: 0)
    weights['left'] = defaultdict(lambda: 0)
    weights['shift'] = defaultdict(lambda: 0)
    data = []
    queque = deque()
    heads = [-1]
    with open('../data/mstparser-en-train.dep', 'r', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                data.append((queque, heads))
                queque = deque()
                heads = [-1]

            else:
                id, word, _, pos, _, _, parent, _ = line.rstrip().split('\t')
                queque.append((int(id), word, pos))
                heads.append(int(parent))

    for _ in tqdm(range(epoch)):
        for queque, heads in data:
            ShiftReduceTrain(queque.copy(), heads, weights)
    with open('model_weights', 'wb') as ans_f:
        dill.dump(weights, ans_f)







