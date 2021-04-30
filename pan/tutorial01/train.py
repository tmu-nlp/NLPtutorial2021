import sys
from collections import defaultdict

def train_unigram(train,model):
  counts = defaultdict(lambda:0)
  total_count = 0
  file = open(train)
  
  for line in file:
    words = line.split()
    words.append('</s>')
    
    for word in words:
      counts[word] += 1
      total_count += 1
      
  model_file = open(model,'w')
  for word, count in sorted(counts.items()):
    prob = counts[word]/total_count
    model_file.write(word + '\t' + str(prob) + '\n')
    
if __name__ == '__main__':
  train_unigram('./data/wiki-ja-train.word')
