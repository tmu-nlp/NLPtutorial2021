from collections import defaultdict
from tqdm import tqdm

class HmmPerceptron():
    def __init__(self):
        self.weights = defaultdict(int)
        self.possible_tags = defaultdict(lambda: 0)
        self.transition = defaultdict(lambda: 0)
    
    def create_possible_labels(self, labels_per_sentence):
        self.possible_tags["<s>"] += 1
        self.possible_tags["</s>"] += 1
        for labels in labels_per_sentence:
            for i in range(len(labels)):
                self.possible_tags[labels[i]] += 1
    
    def create_transition(self, labels_per_sentence):
        for labels in labels_per_sentence:
            prevs, nexts = ["<s>"] + labels, labels + ["</s>"]
            for prv, nxt in zip(prevs, nexts):
                self.transition[f"{prv}_{nxt}"] += 1

    def create_features(self, words, labels):
        phi = defaultdict(lambda: 0)
        for i in range(len(labels)+1):
            first_tag = "<s>" if i == 0 else labels[i-1]
            next_tag = "</s>" if i == len(labels) else labels[i]
            phi[f"T_{first_tag}_{next_tag}"] += 1
        for i in range(len(labels)):
            phi[f"E_{words[i]}_{labels[i]}"] += 1
            # 新しい素性を追加したい 
        return phi

    def update_weights(self, phi_gold, phi_pred):
        for key, val in phi_gold.items():
            self.weights[key] += val
        for key, val in phi_pred.items():
            self.weights[key] -= val
    
    def hmm_viterbi(self, words):
        best_score = defaultdict(lambda: 0)
        best_edge = defaultdict(lambda: 0)
        best_score["0_<s>"] = 0
        best_edge["0_<s>"] = None

        for idx in range(len(words)):
            for prev in self.possible_tags.keys():
                for next in self.possible_tags.keys():
                    if f"{idx}_{prev}" in best_score.keys() and f"{prev}_{next}" in self.transition.keys():
                        score = best_score[f"{idx}_{prev}"] \
                            + self.weights[f"T_{prev}_{next}"] \
                            + self.weights[f"E_{next}_{words[idx]}"]
                        if best_score[f"{idx+1}_{next}"] == 0 or best_score[f"{idx+1}_{next}"] < score:
                            best_score[f"{idx+1}_{next}"] = score
                            best_edge[f"{idx+1}_{next}"] = f"{idx}_{prev}"
                            # best_edge[,_{遷移辞書から考えられる次のタグ}] = 0_<s>
        for label in self.possible_tags.keys():
            if f"{label}_</s>" in self.transition.keys():
                score = best_score[f"{len(words)}_{label}"] + self.weights[f"T_{label}_</s>"]
                if f"{len(words)+1}_</s>" not in best_score or best_score[f"{len(words)+1}_</s>"] < score:
                    best_score[f"{len(words)+1}_</s>"], best_edge[f"{len(words)+1}_</s>"] = score, f"{len(words)}_{prev}"

        labels, next_edge = [], best_edge[f"{len(words)+1}_</s>"]
        
        while next_edge != "0_<s>":
            _, label = next_edge.split("_")
            labels.append(label)
            next_edge = best_edge[next_edge]
        labels = labels[::-1]

        return labels

    def train(self, words_per_sentence, labels_per_sentence, iter=1):
        self.create_possible_labels(labels_per_sentence)
        self.create_transition(labels_per_sentence)
        for _ in tqdm(range(iter)):
            for words, labels in zip(words_per_sentence, labels_per_sentence):
                labels_pred = self.hmm_viterbi(words)
                phi_gold = self.create_features(words, labels)
                phi_pred = self.create_features(words, labels_pred)
                self.update_weights(phi_gold, phi_pred)

    def test(self, words_per_sentence, output_path):
        with open(output_path, "w") as f:
            for words in words_per_sentence:
                labels_pred = self.hmm_viterbi(words)
                f.write(" ".join(labels_pred) + "\n")

def load_file(file_path, is_train=True):
    words_per_sentence, labels_per_sentence = [], []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            words, labels = [], []
            word_label = line.strip().split() 
            for pair in word_label:
                if is_train:
                    word, pos = pair.split("_")
                    words.append(word)
                    labels.append(pos)
                else:
                    word = pair
                    words.append(word)
            words_per_sentence.append(words)
            labels_per_sentence.append(labels)
    return words_per_sentence, labels_per_sentence
    
if __name__ == "__main__":
    train_file = "data/wiki-­en-­train.norm_pos"
    test_file = "data/wiki-­en-­test.norm"
    output_file = "tutorial12/tutorial12.txt"
    train_file = train_file.replace("\xad", "")
    test_file = test_file.replace("\xad", "")

    train_X, train_y = load_file(train_file)
    test_X, _ = load_file(test_file, is_train=False)
    model = HmmPerceptron()
    model.train(train_X, train_y, iter=3)
    model.test(test_X, output_file)