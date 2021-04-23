import sys

def solve(test_text):
    dic_counts = {}

    for line in test_text:
        line = line.strip()
        words = line.split(" ")

        for word in words:
            if word in dic_counts:
                dic_counts[word] += 1
            else:
                dic_counts[word] = 1
                
    list_counts = sorted(dic_counts.items())

    w = open('answer.txt', 'w')

    w.write('\nWord count is {}.\n\n'.format(len(list_counts)))
    for word, count in list_counts:
        w.write('{}    {}\n'.format(word, count))
    
    w.close()
    return 0


test = open(sys.argv[1], "r")

solve(test)