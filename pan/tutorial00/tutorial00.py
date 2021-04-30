import sys

counts = {}
file = open(sys.argv[1])

words = []
for line in file:  
  words = line.split()
    
  for word in words:
      if word in counts:
          counts[word] +=1
      else:
          counts[word] = 1

for x, y in sorted(counts.items()):
  print ('%s: %r' % (x, y))

print('\nNumber of Unique Words: ' + str(len(counts)))



#def wordcount(text):
#  list = text.split()
#  dict = {}
#  for word in list:
#    if word in dict:
#      dict[word] += 1
#    else:
#      dict[word[ = 1
#  return dict
                
                
                
                
#process         
#1#create an empty dict
#2#save every word as a key to the dict and set its value to 1
#3#circle from beginning to the end
#4#if the key of the word already exists add 1
