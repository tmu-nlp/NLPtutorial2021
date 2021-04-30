from collections import defaultdict

def UnigramModel(train_file):
  dic = defaultdict(lambda: 0)
  total_count = 0

  with open(train_file) as f:
    for line in f:
      words = line.split()
      words.append("</s>")
      for word in words:
        dic[word] += 1
        total_count += 1

  model_file = open("tutorial01.txt", "w")
  for word in dic.keys():
    prob = dic[word] / total_count
    model_file.write(f"{word} {prob}" + "\n")

if __name__ == "__main__":
  #data/wiki-en-train.word
  train_file = "/content/drive/MyDrive/nlptutorial-master/data/wiki-en-train.word"
  UnigramModel(train_file)