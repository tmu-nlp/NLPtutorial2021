from collections import defaultdict
import math

w = defaultdict(lambda: 0)
last = defaultdict(lambda: 0)
margin = 20
c = 0.0001
alpha = 4
count = 0

def getw(w, name, c, iter, last):
    if iter != last[name]:
        c_size = c * (iter - last[name])
        if abs(w[name]) <= c_size:
            w[name] = 0
        else:
            w[name] -= sign(w[name]) * c_size
        last[name] = iter
    return w[name]

def sign(a):
    if a >= 0:
        return 1
    else:
        return -1

def create_features(x):
    phi = defaultdict(lambda: 0)
    words = x.split(' ')
    for word in words:
        phi['UNI:' + word] += 1
    return phi

def update_weights(w, phi, y):
    wphi = 0
    for name, value in phi.items():
        if name in w:
            wphi += w[name] * value
    for name, value in phi.items():
        w[name] = getw(w, name, c, count, last)
        w[name] += value * y
        #w[name] += y * alpha * value * (math.exp(wphi) / (1 + math.exp(wphi))**2)

with open('/users/kcnco/github/NLPtutorial2021/pan/tutorial06/titles-en-train.labeled', 'r') as input_file, open('/users/kcnco/github/NLPtutorial2021/pan/tutorial06/model-file.txt', 'w') as model_file:
    for line in input_file:
        line = line.strip().split('\t')
        x = line[1]
        y = int(line[0])
        phi = create_features(x)
        val = 0
        for name, value in phi.items():
            if name in w:
                val += w[name] * value
        val = val * y
        if val <= margin:
            count += 1
            update_weights(w, phi, y)
    for name, value in w.items():
        w[name] = getw(w, name, c, count, last);
    for name, value in sorted(w.items()):
        print(f'{name} {value}', file=model_file)
