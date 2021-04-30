from collections import defaultdict
import math

def Test(model_file, test_file):
  lambda_1=0.95; lambda_unk=1-lambda_1; V=10**6; W=0; H=0; cnt_unk=0
  
  #モデル読み込み
  probs = defaultdict(lambda: 0)
  with open(model_file) as f:
    for line in f:
      word, prob = line.split()
      probs[word] = prob

  #評価と結果表示
  with open(test_file) as f:
    for line in f:
      words = line.split()
      words.append("</s>")
      for word in words:
        W += 1
        P = lambda_unk / V
        if word in probs:
          P += lambda_1 * float(probs[word])
        else:
          cnt_unk += 1
        H += -math.log(P, 2)
  
  print(f"Entropy: {H/W}")
  print(f"Coverage: {(W-cnt_unk)/W}")

if __name__ == "__main__":
  model_file = "tutorial01.txt"
  #data/wiki-en-test.word
  test_file = "/content/drive/MyDrive/nlptutorial-master/data/wiki-en-test.word"
  Test(model_file, test_file)