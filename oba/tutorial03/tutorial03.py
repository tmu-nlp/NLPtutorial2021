import sys
import os
sys.path.append("../")
from tutorial01.tutorial01 import UnigramLangModel

class Tokenizer(UnigramLangModel):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    tokenizer = Tokenizer()

    # train
    train_file = "../data/wiki-ja-train.word"
    model_file = "tutorial03.txt"
    tokenizer.train(train_file_pth=train_file).save(prob_file=model_file)