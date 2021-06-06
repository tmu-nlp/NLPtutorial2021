from collections import defaultdict
from train_perceptron import *
import os

if __name__ == '__main__':
    model = open('titles-en-train.labeled', 'r')
    input = open('titles-en-test.word', 'r')
    w = train_perceptron(10, model)
    
    predict_all(w, input)
    os.system('python grade-prediction.py titles-en-test.labeled 05-answer.labeled')

    model.close()
    input.close()

# $ python grade-prediction.py titles-en-test.labeled 05-answer.labeled
# Accuracy = 90.860786%