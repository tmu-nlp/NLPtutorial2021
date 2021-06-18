import numpy as np
from collections import defaultdict

class RNN():
    def __init__(self, word_to_id, tag_to_id, lr=0.01, hidden_size=64) -> None:
        self.word_to_id = word_to_id
        self.tag_to_id = tag_to_id
        self.lr = lr
        self.network = []
        self.hidden_size = hidden_size
        self.embed_size = len(word_to_id)
        self.output_size = len(tag_to_id)
        
        # rand：-0.5以上, 0.5未満 関数として定義すべき？
        self.w_rx = np.random.rand(self.hidden_size, self.embed_size)/5 - 0.1
        self.w_rh = np.random.rand(self.hidden_size, self.hidden_size)/5 - 0.1
        self.b_r = np.random.rand(self.hidden_size)/5 - 0.1
        self.w_oh = np.random.rand(self.output_size, self.hidden_size)/5 - 0.1
        self.b_o = np.random.rand(self.output_size)/5 - 0.1
    
    def id_to_onehot(self, ids, to_id):
        onehot = np.zeros([len(ids), len(to_id)])
        for i, id in enumerate(ids):
            onehot[i][id] += 1
        return onehot
    
    def find_max(self, prob):
        p_max = 0
        for p in prob:
            if p > p_max:
                p_max = p
        return p_max

    def forward(self, X):# X：1sentence分のid
        self.hidden, self.prob, self.y_pred, self.input= [], [], [], []
        for time in range(len(X)):
            if time > 0:
                self.hidden.append(np.tanh(np.dot(self.w_rx, X[time]) + np.dot(self.w_rh, self.hidden[time-1]) + self.b_r))
            else:
                self.hidden.append(np.tanh(np.dot(self.w_rx, X[time]) + self.b_r))
            
            self.prob.append(np.tanh(np.dot(self.w_oh, self.hidden[time]) + self.b_o))
            self.y_pred.append(self.find_max(self.prob[time]))
            self.input.append(X[time])
        return self.y_pred

    def gradient(self, y_gold):
        self.delta_w_rx = np.zeros(self.w_rx.shape)
        self.delta_w_rh = np.zeros(self.w_rh.shape)
        self.delta_b_r = np.zeros(self.b_r.shape)
        self.delta_w_oh = np.zeros(self.w_oh.shape)
        self.delta_b_o = np.zeros(self.b_o.shape)
        delta_r_prime = np.zeros(self.hidden_size)
        for i in range(1, len(y_gold)+1):
            t = len(y_gold) - i
            delta_o_prime = y_gold[t] - self.y_pred[t]
            self.delta_w_oh += np.outer(self.hidden[t], delta_o_prime).T
            self.delta_b_o += delta_o_prime
            delta_r = np.dot(delta_r_prime, self.w_rh) + np.dot(delta_o_prime, self.w_oh)
            delta_r_prime = delta_r * (1-self.hidden[t]**2)
            self.delta_w_rx += np.outer(self.input[t], delta_r_prime).T
            self.delta_b_r += delta_r_prime
            if t!=0:
                self.delta_w_rh += np.outer(self.hidden[t-1], delta_r_prime).T
        return self

    def update_weights(self):
        self.w_rx += self.lr*self.delta_w_rx
        self.w_rh += self.lr*self.delta_w_rh
        self.b_r += self.lr*self.delta_b_r
        self.w_oh += self.lr*self.delta_w_oh
        self.b_o += self.lr*self.delta_b_o
        return self

    def train(self, X_id, y_id, iter=10):
        for i in range(iter):
            for i in range(len(X_id)):
                X_vec = self.id_to_onehot(X_id[i], self.word_to_id)
                y_vec = self.id_to_onehot(y_id[i], self.tag_to_id)
                y_pred = self.forward(X_vec)
                self.gradient(y_vec)
                self.update_weights()
    
    def predict(self, X_id):
        predictions = []
        for i in range(len(X_id)):
            X_vec = self.id_to_onehot(X_id[i], self.word_to_id)
            y_pred = self.forward(X_vec)
            predictions.append(y_pred)
        print(y_pred)
        return predictions
            
                
def load_data(file_pth, is_train=True, has_word_to_id=None):
        if has_word_to_id:
            word_to_id = has_word_to_id
        else:
            word_to_id = defaultdict(lambda: len(word_to_id))
        tag_to_id = defaultdict(lambda: len(tag_to_id))
        word_to_id["<UNK>"]
        sentences, tags = [], []
        
        with open(file_pth, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if is_train:
            for line in lines:
                word_ids, tag_ids = [], []
                for word_tag in line.strip().split():
                    word, tag = word_tag.split("_")
                    word_ids.append(word_to_id[word]), tag_ids.append(tag_to_id[tag])
                sentences.append(word_ids), tags.append(tag_ids)
        else:
            for line in lines:
                word_ids = []
                for word in line.strip().split():
                    if word in word_to_id.keys():
                        word_ids.append(word_to_id[word])
                    else:
                        word_ids.append(word_to_id["<UNK>"])
                sentences.append(word_ids)
        return sentences, tags, word_to_id, tag_to_id
        #[id, id, ...], [word:id, word:id, ...]


if __name__ == "__main__":
    train_file = "data/wiki-en-train.norm_pos"
    test_file = "data/wiki-en-test.norm"
    train_X, train_y, word_to_id, tag_to_id = load_data(file_pth=train_file)
    model = RNN(word_to_id=word_to_id, tag_to_id=tag_to_id)
    model.train(train_X, train_y)

    test_X, _, _, _ = load_data(file_pth=test_file, is_train=False, has_word_to_id=word_to_id)
    model.predict(test_X)