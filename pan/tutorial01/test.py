import sys
import math

def test_unigram(model,test):
  model_file = open(model)
  test_file = open(test)
  prob = defaultdict(lambda:0)

  lambda_1 = 0.95
  lambda_unk = 1 - lambda_1
  V = 1000000
  W = 0
  H = 0
  unknown_word = 0

  for line in model_file:
      words = line.split()
      w = words[0]
      P = words[1]
      prob[w] = float(P)

  for line in test_file:
      words = line.split()
      words.append('</s>')

      for w in words:
          W += 1
          P = lambda_unk / V

          if w in prob:
              P += lambda_1 * prob[w]
          else:
              unknown_word += 1
          H += -math.log2(P)
            
  print('Entropy = ' + str(round(H / W, 6)))
  print('Coverage = ' + str(round((W - unknown_word) / W, 6)))

if __name__ == '__main__':
  test_unigram(sys.argv[1],sys.argv[2])
