from collections import defaultdict
import numpy as np
import pickle

def create_features(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        phi[ids["UNI:" + word]] += 1
    return phi


def init_network(feature_size, node, layer):
    net = []

    # 入力層
    w0 = 2 * np.random.rand(node, feature_size) - 0.5
    b0 = np.random.rand(1, node)
    net.append((w0, b0))

    # 中間層
    while len(net) < layer:
        w = 2 * np.random.rand(node, node) - 0.5
        b = np.random.rand(1, node)
        net.append((w, b))

    # 出力層
    w_o = 2 * np.random.rand(1, node) - 0.5
    b_o = np.random.rand(1, 1)

    net.append((w_o, b_o))

    return net


def forward_nn(net, phi_0):
    phi = [0 for _ in range(len(net) + 1)]  
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T 
    return phi


def backward_nn(net, phi, label):
    j = len(net)
    delta = np.zeros(j + 1, dtype=np.ndarray) 
    delta[-1] = np.array([label - phi[j][0]])
    delta_d = np.zeros(j + 1, dtype=np.ndarray)
    for i in range(j, 0, -1):
        delta_d[i] = delta[i] * (1 - phi[i] ** 2).T  
        w, _ = net[i - 1]
        delta[i - 1] = np.dot(delta_d[i], w)
    return delta_d


def update_weights(net, phi, delta_d, eta):
    for i in range(len(net)):
        w, b = net[i]
        w += eta * np.outer(delta_d[i + 1], phi[i])
        b += eta * delta_d[i + 1]

def predict_one(net, phi_0):
    phi = [0 for _ in range(len(net) + 1)]
    phi[0] = phi_0
    for i in range(len(net)):
        w, b = net[i]
        phi[i + 1] = np.tanh(np.dot(w, phi[i]) + b).T
    score = phi[len(net)][0]
    return 1 if score >= 0 else -1


def create_features_test(sentence, ids):
    phi = np.zeros(len(ids))
    words = sentence.split()
    for word in words:
        if "UNI:" + word not in ids:
            continue
        phi[ids["UNI:" + word]] += 1

    return phi

np.random.seed(seed=0)
ids = defaultdict(lambda: len(ids))
feat_label = []

with open("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-train.labeled", "r", encoding="utf-8") as train_file:
    for line in train_file:
        label, sentence = line.strip().split("\t")
        for word in sentence.split():
            ids["UNI:" + word]

with open("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-train.labeled", "r", encoding="utf-8") as train_file:
    for line in train_file:
        label, sentence = line.strip().split("\t")
        label = int(label)
        phi = create_features(sentence, ids)
        feat_label.append((phi, label))

net = init_network(len(ids), 2, 1)

for _ in range(5):
    for phi_0, label in feat_label:
        phi = forward_nn(net, phi_0)
        delta_d = backward_nn(net, phi, label)
        update_weights(net, phi, delta_d, 0.1)


with open("net", "wb") as net_file, open("ids", "wb") as ids_file:
    pickle.dump(net, net_file)
    pickle.dump(dict(ids), ids_file)


np.random.seed(seed=0)
with open("net", "rb") as net_file, open("ids", "rb") as ids_file:
    net = pickle.load(net_file)
    ids = pickle.load(ids_file)

with open("/Users/lingzhidong/Documents/GitHub/nlptutorial/data/titles-en-test.word", "r", encoding="utf-8") as test_file, open("my_result.txt", "w", encoding="utf-8") as ans_file:
    for line in test_file:
        phi = create_features_test(line.strip(), ids)
        predict = predict_one(net, phi)
        ans_file.write(str(predict) + "\t" + line)

