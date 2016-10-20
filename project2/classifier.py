import os
import sys
from collections import Counter

TRAIN_DIR = './train'
TEST_PUBLIC_DIR = './test-public'
TEST_PRIVATE_DIR = './test-private'

TRANSITION_DATA = dict()
TOTAL_TAG_COUNT = '*'
WORD_BIO_LIST = []

def generate_probs():
  # prevents key presence checking later on (only 9 possible BIO transitions exist)
  TRANSITION_DATA['B'] = dict()
  TRANSITION_DATA['B']['B'] = 0
  TRANSITION_DATA['B']['I'] = 0
  TRANSITION_DATA['B']['O'] = 0
  TRANSITION_DATA['I'] = dict()
  TRANSITION_DATA['I']['B'] = 0
  TRANSITION_DATA['I']['I'] = 0
  TRANSITION_DATA['I']['O'] = 0
  TRANSITION_DATA['O'] = dict()
  TRANSITION_DATA['O']['B'] = 0
  TRANSITION_DATA['O']['I'] = 0
  TRANSITION_DATA['O']['O'] = 0

  for dir_name, sub_dir_list, file_list in os.walk(TRAIN_DIR):
    # ignore hidden files
    file_list = [f for f in file_list if f[0] != '.']
    test_list = ['test.txt']

    for file_name in file_list:
      file_path = os.path.join(TRAIN_DIR,file_name)
      file_content = open(file_path)
      old_cue_value = 0 # start with state of not having seen any CUE's yet
      last_bio = 'O'
      last_line = ''

      for line in iter(file_content):
        # don't count empty lines (end of sentence market)
        if line != '\n':
          line_values = line.split()
          word = line_values[0]
          cue = line_values[2] # e.g. CUE-1 or CUE-2 ...

          # must then be a CUE-X, so keep track of the value of X
          if cue != '_':
            new_cue_value = int(cue[-1])
            if old_cue_value == new_cue_value:
              TRANSITION_DATA[last_bio]['I'] += 1
              last_bio = 'I'
              last_line = line
              # I in BIO
              WORD_BIO_LIST.append((word, 'I'))
            
            if old_cue_value == new_cue_value - 1:
              TRANSITION_DATA[last_bio]['B'] += 1
              last_bio = 'B'
              last_line = line
              # B in BIO
              WORD_BIO_LIST.append((word, 'B'))
            
            # update this value for the next token
            old_cue_value = new_cue_value
          
          # O in BIO format
          else:
            TRANSITION_DATA[last_bio]['O'] += 1
            last_bio = 'O'
            last_line = line
            # O in BIO format
            WORD_BIO_LIST.append((word, 'O'))

        # it is an empty line, so clear the BIO count state
        else:
          old_cue_value = 0 # all CUE-X's are cleared across sentences
          last_bio = 'O' # this will only marginally affect the count
          last_line = line

def classify_tokens_and_sentences():
  EMISSION_COUNTS = Counter(WORD_BIO_LIST)
  _, bio_list = zip(*WORD_BIO_LIST)
  TOTAL_BIO_COUNTS = Counter(bio_list)
  token_bio_classifications = []
  
  for dir_name, sub_dir_list, file_list in os.walk(TEST_PUBLIC_DIR):
    # ignore hidden files
    file_list = [f for f in file_list if f[0] != '.']
    test_list = ['test.txt']

    for file_name in test_list:
      file_path = os.path.join(TEST_PUBLIC_DIR,file_name)
      file_content = open(file_path)
      last_bio = 'O'

      for line in iter(file_content):
        if line != '\n':
          line_values = line.split()
          word = line_values[0]
          print word
          
          # compute P(word | tag) * P(this_tag | last_tag) for each bio tag
          computed_bio_products = []
          for tag in ['B', 'I', 'O']:
            p_word_given_tag = EMISSION_COUNTS[(word, tag)] / float(TOTAL_BIO_COUNTS[tag])
            p_transition = TRANSITION_DATA[last_bio][tag] / float(TOTAL_BIO_COUNTS[last_bio])
            product = p_word_given_tag*p_transition
            print product
            computed_bio_products.append((product, tag))

          max_prob_tag = max(computed_bio_products, key=lambda x:x[0])
          token_bio_classifications.append(max_prob_tag[1])
        
        else:
          # it is an empty line, so mark this in the growing list of tags
          # with a sentence delimiter '*'
          token_bio_classifications.append('*')

        print '\n'

    print token_bio_classifications


def run():
  generate_probs()
  classify_tokens_and_sentences()

if __name__=='__main__':
  run()
