import random
import math
from tqdm import tqdm
from collections import defaultdict
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess(text):
  text = re.sub('[^a-zA-Z]+',' ', text)
  text = [i for i in text.split() if not i in stop_words] # remove stopwords
  return text

def SampleOne(probs):
    z = sum(probs)
    remaining = random.uniform(0, z)
    for i in range(len(probs)):
        remaining = remaining - probs[i]
        if remaining <= 0:
            return i

class lda():
    def __init__(self, num_topics = 2):
        self.num_topics = num_topics
        self.xcorpus = []
        self.ycorpus = []
        self.vocab = defaultdict(int)
        self.xcounts = defaultdict(int)
        self.ycounts = defaultdict(int)

    def init(self,path):
        with open(path,'r') as f:
            for line in f:
                words=preprocess(line.strip())
                topics=[]
                for word in words:
                    topic=random.randint(1,self.num_topics)
                    topics.append(topic)
                    self.vocab[word]+=self.vocab[word]+1
                    self.add_counts(word, topic, len(self.xcorpus), 1)
                self.xcorpus.append(words)
                self.ycorpus.append(topics)

    
    def add_counts(self,word,topic,docid,amount):
        self.xcounts[topic] += amount
        self.xcounts[word + "|" + str(topic)] += amount
        self.ycounts[docid] += amount
        self.xcounts[str(topic) + "|" + str(docid)] += amount

    def TrainLda(self, epoch, alpha=0.1, beta=0.1):
        for _ in range(epoch):
            ll = 0
            for i in tqdm(range(len(self.xcorpus))):
                for j in range(len(self.xcorpus[i])):
                    x = self.xcorpus[i][j]
                    y = self.ycorpus[i][j]
                    self.add_counts(x, y, i, -1)
                    probs = []
                    for k in range(self.num_topics):
                        p_x_k = (self.xcounts[x +"|"+ str(k)]+alpha)/(self.ycounts[k]+alpha*len(self.vocab))
                        p_k_y = (self.ycounts[str(k) +"|"+ str(y)]+beta)/(self.ycounts[y]+beta*self.num_topics)
                        probs.append(p_x_k*p_k_y)

                    new_y = SampleOne(probs)
                    ll += math.log(probs[new_y])
                    self.add_counts(x, new_y, i, 1)
                    self.ycorpus[i][j] = new_y

file_path="/Users/lingzhidong/Documents/GitHub/nlptutorial/data/wiki-en-documents.word"

ldaModel = lda()
ldaModel.init(file_path) 
ldaModel.TrainLda(5)

topic1=defaultdict(int)
topic2=defaultdict(int)
for words, topics in zip(ldaModel.xcorpus, ldaModel.ycorpus):
    for word, topic in zip(words, topics):
        if topic == 0:
            topic1[word] += 1
        else:
            topic2[word] += 1

print("TopicA")
print(sorted(topic1.items(), key=lambda x: x[1], reverse=True)[:100])
print("TopicB")
print(sorted(topic2.items(), key=lambda x: x[1], reverse=True)[:100])


'''
TopicA
[('could', 113), ('ward', 94), ('founder', 92), ('kannon', 66), ('scale', 50), ('middle', 44), ('otokuni', 44), ('track', 43), ('states', 42), ('takarazuka', 42), ('yoshitsune', 41), ('printed', 39), ('government', 39), ('yoshio', 38), ('wooden', 37), ('tenryu', 36), ('degree', 36), ('kamomioya', 36), ('became', 35), ('agreement', 35), ('taisho', 34), ('key', 34), ('sendai', 33), ('thousand', 33), ('gu', 32), ('rights', 32), ('deities', 32), ('live', 32), ('rapid', 32), ('seasons', 31), ('zen', 31), ('historical', 31), ('similar', 31), ('fish', 31), ('results', 30), ('actor', 30), ('throne', 30), ('arrived', 29), ('sixth', 29), ('aizen', 28), ('average', 28), ('akamatsu', 28), ('sciences', 28), ('know', 27), ('start', 27), ('bow', 27), ('grain', 27), ('uta', 27), ('nihongi', 27), ('sanin', 27), ('refused', 26), ('esoteric', 26), ('buried', 26), ('governor', 26), ('ujo', 26), ('kanrei', 26), ('ogimachi', 25), ('decision', 25), ('keihanna', 25), ('michizane', 25), ('shizuoka', 25), ('whereas', 24), ('ono', 24), ('royal', 24), ('field', 24), ('tokai', 23), ('asia', 23), ('content', 23), ('ceremony', 23), ('suiboku', 23), ('cars', 23), ('torii', 23), ('letter', 22), ('komyo', 22), ('nakayama', 22), ('deity', 22), ('naka', 22), ('international', 22), ('choice', 22), ('adult', 22), ('za', 22), ('surrender', 22), ('nagaokakyo', 22), ('ritsumeikan', 22), ('children', 21), ('beings', 21), ('failed', 21), ('lie', 21), ('containing', 21), ('less', 21), ('tei', 21), ('mud', 21), ('end', 21), ('signed', 21), ('hoshi', 21), ('assassinated', 21), ('crossing', 21), ('reassigned', 21), ('tramline', 21), ('camp', 20)]
TopicB
[('apos', 16760), ('quot', 10460), ('station', 5680), ('kyoto', 5668), ('temple', 3685), ('emperor', 3648), ('line', 3566), ('city', 3298), ('school', 2726), ('period', 2669), ('ji', 2504), ('also', 2476), ('imperial', 2440), ('family', 2234), ('one', 2073), ('prefecture', 1885), ('called', 1882), ('japan', 1764), ('shrine', 1715), ('clan', 1702), ('became', 1682), ('university', 1632), ('name', 1564), ('time', 1544), ('however', 1415), ('japanese', 1377), ('used', 1366), ('trains', 1355), ('railway', 1338), ('first', 1333), ('court', 1328), ('province', 1234), ('established', 1210), ('fujiwara', 1186), ('main', 1169), ('train', 1167), ('government', 1141), ('two', 1136), ('section', 1128), ('year', 1110), ('national', 1047), ('system', 1027), ('prince', 1022), ('many', 1021), ('son', 1019), ('later', 1018), ('made', 1012), ('said', 1002), ('express', 1000), ('people', 998), ('area', 993), ('street', 986), ('old', 944), ('new', 925), ('osaka', 916), ('since', 883), ('three', 860), ('edo', 853), ('castle', 848), ('order', 833), ('war', 830), ('cultural', 822), ('years', 804), ('hall', 804), ('th', 787), ('ward', 785), ('dori', 781), ('shogun', 769), ('west', 768), ('day', 759), ('april', 759), ('around', 752), ('style', 746), ('cho', 745), ('kamakura', 744), ('known', 739), ('well', 735), ('high', 735), ('sect', 726), ('head', 722), ('go', 718), ('shogunate', 708), ('keihan', 707), ('rank', 704), ('local', 702), ('river', 696), ('jr', 696), ('maizuru', 696), ('domain', 692), ('second', 691), ('side', 676), ('located', 676), ('power', 669), ('jinja', 666), ('ashikaga', 663), ('buddhist', 661), ('written', 656), ('department', 653), ('important', 646), ('built', 645)]
'''