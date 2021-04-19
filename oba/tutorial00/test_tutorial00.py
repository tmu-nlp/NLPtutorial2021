import unittest
import sys
from tutorial00 import word_count

class WordCountTest(unittest.TestCase):
    def test_word_count(self) :
        sys.argv.append("../data/00-input.txt")
        sys.argv.append("../data/00-answer.txt")
        input_path = sys.argv[1]
        answer_path = sys.argv[2]
        
        test_count_list = word_count(input_path)

        with open(answer_path) as ans_file:
            answer = ans_file.readlines()
            answer = [line.strip().split("\t") for line in answer]
            answer_count_list = [(pair[0], int(pair[1])) for pair in answer] 
            self.assertEqual(test_count_list == answer_count_list, True)

if __name__ == "__main__":
    # test = WordCountTest()
    # test_result = test.test_word_count()
    
    unittest.main()
    