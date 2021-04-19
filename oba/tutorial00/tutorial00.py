import sys
from collections import defaultdict

def word_count(file_path):
    # デフォルトを任意の値に設定
    count = defaultdict(lambda: 0)

    with open(file_path) as file:
        for sentence in file:
            for word in sentence.split():
                count[word] += 1

    count_list = list(count.items())
    return count_list

if __name__ == "__main__":
    file_path = sys.argv[1]
    # file_path = "../data/wiki-en-train.word"

    word_counter = word_count(file_path=file_path)
    word_tab_count = [f"{word}\t{count}" for (word, count) in word_counter]
    print("\n".join(word_tab_count))