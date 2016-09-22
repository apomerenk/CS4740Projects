import os
import nltk
import glob
import sys
import numpy
import math

"""
NOTES ON RUNNING THE LANGUAGE MODELS:

In order to run our unigram and bigram language models, you will need to have
installed the Natural Language Toolkit (nltk) through pip. The relevant CL commands
are inside of setup.py. Once installed, you will need to download the corpora used
for tokenizing within nltk by entering a python shell and entering the following commands:
  > import nltk
  > nltk.download()
From here, a new Python application will launch and prompt you to download the relevant
nltk packages. Select and download all packages (~ 5 minutes). 

In addition, you will need to have numpy installed in order for us to correctly sample
our probability distributions for each language model.

EXAMPLE USAGES
To generate random sentences from the command line from the root directory labeled 'project1', run:
  >python model.py random_sentence <corpus_to_train_on>
  >python model.py random_sentence autos

If insufficient arguments are supplied to the application, it will prompt for the
relevant command line arguments. Running:
  >python model.py test
will run a quick test training on the TEST_EMAIL constant below

Enjoy! :)
"""

TEST_ROOT_DIRECTORY = './atheism'
ROOT_DIRECTORY = './data_corrected/classification task/'
TEST_CLASSIFICATION_DIRECTORY = './data_corrected/classification task/test_for_classification'
TEST_DIR_NAME = 'test_for_classification'
PUNCT_TO_REMOVE = ['(', ')', '<', '>', ',', '/', '-', '"', 
                   ':', '``', '*', '[', ']', '#', '|', "'", '$', '%',
                   '^', '~', "`", "\\", '_', '+', '{', '}', '=']
UNIGRAM_COUNT_SYM = '*'
LANGUAGE_MODEL = dict()
# mapping from (<bigram appearance count>, <total number of bigrams with that count>)
BIGRAM_N_COUNTS = dict()
CORPUS_WORD_COUNT = 0
UNK_WORD_COUNT = 0
UNK_WORD_SET = set()
VOCAB = set()
SENTENCE_ENDS = ['?', '!', '.']
UNK_WORD = "unk"

TEST_EMAIL = "the dog jumped over the log"

def test_sample_string(test):
  tokens = nltk.word_tokenize(test)
  processed_tokens = preprocess_file_tokens(tokens)
  update_language_model(processed_tokens)
  print LANGUAGE_MODEL

def update_language_model(tokens):
  global CORPUS_WORD_COUNT
  global UNK_WORD_COUNT
  global VOCAB
  for i in range(len(tokens)-1):
    # keep track of total words in the corpus
    CORPUS_WORD_COUNT += 1
    # make dict keys consistent
    token = tokens[i].lower()
    next_token = tokens[i+1].lower()

    if token in LANGUAGE_MODEL:
      if token in UNK_WORD_SET:
        # we have now seen the token twice
        UNK_WORD_COUNT -= 1
        UNK_WORD_SET.remove(token)
      # increment unigram count of string
      LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM] += 1
      # increment bigram count of 2-element sequence
      if next_token in LANGUAGE_MODEL[token]:
        LANGUAGE_MODEL[token][next_token] += 1
      else:
        LANGUAGE_MODEL[token][next_token] = 1
    else:
      # new word, so record it as an <unk> to start
      UNK_WORD_COUNT += 1
      UNK_WORD_SET.add(token)
      # add new token to vocabulary
      VOCAB.add(token)
      LANGUAGE_MODEL[token] = dict()
      # set first-time unigram count
      LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM] = 1
      # set first-time bigram count
      LANGUAGE_MODEL[token][next_token] = 1

  # edge case of final string in token list (only update unigram count)
  CORPUS_WORD_COUNT += 1
  final_token = tokens[-1]
  # subtract out 1-time appearing words from the VOCAB set
  VOCAB -= UNK_WORD_SET
  VOCAB.add(UNK_WORD)

  if final_token in LANGUAGE_MODEL:
    LANGUAGE_MODEL[final_token][UNIGRAM_COUNT_SYM] += 1
  else:
    # new word, so record it as an <unk> to start
    UNK_WORD_COUNT += 1
    UNK_WORD_SET.add(final_token)
    LANGUAGE_MODEL[final_token] = dict()
    LANGUAGE_MODEL[final_token][UNIGRAM_COUNT_SYM] = 1

def modify_unknown_words():
  # aggregate outer unk words into one key
  LANGUAGE_MODEL[UNK_WORD] = dict()
  for token, token_dict in LANGUAGE_MODEL.items():
    # if in set, it only appeared once in the corpus
    if token in UNK_WORD_SET:
      for second_token, bigram_count in token_dict.items():
        if second_token != UNIGRAM_COUNT_SYM:
          if second_token in LANGUAGE_MODEL[UNK_WORD]:
            del LANGUAGE_MODEL[token]
            LANGUAGE_MODEL[UNK_WORD][second_token] += bigram_count
          else:
            del LANGUAGE_MODEL[token]
            LANGUAGE_MODEL[UNK_WORD][second_token] = bigram_count
  
  # lower layer unk word replacement
  for token, token_dict in LANGUAGE_MODEL.items():
    # add an entry for any token followed by UNK (token, UNK) 
    if UNK_WORD not in LANGUAGE_MODEL[token]:
      LANGUAGE_MODEL[token][UNK_WORD] = 0
    for second_token, bigram_count in token_dict.items():
      if second_token != UNIGRAM_COUNT_SYM and second_token in UNK_WORD_SET:
        # remove the existing entry and replace it with UNK as the next_token
        del LANGUAGE_MODEL[token][second_token]
        LANGUAGE_MODEL[token][UNK_WORD] += bigram_count


def preprocess_file_tokens(tokens):
  # removes extraneous punctuation
  filtered_tokens = filter(lambda token: token not in PUNCT_TO_REMOVE, tokens)
  # remove email users and domains
  indexes_to_remove = []
  for i in range(len(filtered_tokens)):
    if filtered_tokens[i] == '@':
      indexes_to_remove.extend((i-1, i, i+1))
  for i in reversed(indexes_to_remove):
    del filtered_tokens[i]
  return filtered_tokens

def tokenize_files(root = TEST_ROOT_DIRECTORY):
  for dir_name, sub_dir_list, file_list in os.walk(root):
    if dir_name != TEST_DIR_NAME:
      # selectively filter out hidden files
      file_list = [f for f in file_list if f[0] != '.']
      for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        file_content = open(file_path).read()
        # remove contractions for tokenizing
        file_content = file_content.replace("'", "")
        tokens = nltk.word_tokenize(file_content)
        processed_tokens = preprocess_file_tokens(tokens)
        # update unigram and bigram counts for all tokens
        update_language_model(processed_tokens)

def generate_unigram_probability_distribution():
  word_list = []
  probability_list = []
  for token, token_dict in LANGUAGE_MODEL.iteritems():
    word_list.append(token)
    probability_list.append(token_dict[UNIGRAM_COUNT_SYM] / float(CORPUS_WORD_COUNT))
  return word_list, probability_list

def unigram_random_sentence(words, probabilities):
  sentence_list = []
  first_word = numpy.random.choice(words, p=probabilities)
  sentence_list.append(first_word.title())
  while sentence_list[-1] not in SENTENCE_ENDS:
    next_word = numpy.random.choice(words, p=probabilities)
    sentence_list.append(next_word)
  return convert_list_to_string(sentence_list)

def bigram_random_sentence():
  sentence_list = []
  aggregate_count = LANGUAGE_MODEL['?'][UNIGRAM_COUNT_SYM] + LANGUAGE_MODEL['!'][UNIGRAM_COUNT_SYM] + LANGUAGE_MODEL['.'][UNIGRAM_COUNT_SYM]
  marker_probs = [LANGUAGE_MODEL['?'][UNIGRAM_COUNT_SYM]/float(aggregate_count), LANGUAGE_MODEL['!'][UNIGRAM_COUNT_SYM]/float(aggregate_count), LANGUAGE_MODEL['.'][UNIGRAM_COUNT_SYM]/float(aggregate_count)]
  sentence_marker = numpy.random.choice(SENTENCE_ENDS, p=marker_probs)
  
  possible_next_words = LANGUAGE_MODEL[sentence_marker].items()
  possible_next_words = [(token, count) for token, count in possible_next_words if token != UNIGRAM_COUNT_SYM]
  sentence_marker_count = sum([count for token, count in possible_next_words])
  next_word_probs = []
  next_words = []
  for token, count in possible_next_words:
    next_word_probs.append(count/float(sentence_marker_count))
    next_words.append(token)
  first_word = numpy.random.choice(next_words, p=next_word_probs)
  sentence_list.append(first_word.title())
  last_word = first_word
  
  while sentence_list[-1] not in SENTENCE_ENDS:
    possible_next_words = LANGUAGE_MODEL[last_word].items()
    possible_next_words = [(token, count) for token, count in possible_next_words if token != UNIGRAM_COUNT_SYM]
    next_word_count = sum([count for token, count in possible_next_words])
    next_word_probs = []
    next_words = []
    for token, count in possible_next_words:
      next_word_probs.append(count/float(next_word_count))
      next_words.append(token)
    # edge case where there is no sentence terminator
    if len(next_words) == 0:
      sentence_list.append('.')
      return convert_list_to_string(sentence_list)
    next_word = numpy.random.choice(next_words, p=next_word_probs)
    sentence_list.append(next_word)
    last_word = next_word
  return convert_list_to_string(sentence_list)

def convert_list_to_string(lst):
  sentence = ""
  if len(lst) <= 1:
    return lst[0]
  elif len(lst) == 2:
    return lst[0]+lst[1]
  else:
    for word in lst[:-2]:
      sentence += word
      sentence += " "
    return sentence + lst[-2] + lst[-1]

# aggregate the counts for all N(0), N(1), N(2), etc... that represent the number
# of bigrams that appear in the corpus with total count of 1, 2, 3, etc...
def compute_total_bigram_counts():
  number_of_bigrams = 0
  for token, token_dict in LANGUAGE_MODEL.iteritems():
    for second_token, bigram_count in token_dict.iteritems():
      # ignore the unigram count symbol and bigram counts greater than 10
      if second_token != UNIGRAM_COUNT_SYM: 
        # count total number of bigrams
        if bigram_count > 0:
          number_of_bigrams += 1
        if bigram_count <= 10:
          if bigram_count in BIGRAM_N_COUNTS:
            BIGRAM_N_COUNTS[bigram_count] += 1
          else:
            BIGRAM_N_COUNTS[bigram_count] = 1
  
  # finally, compute N(0) = |V|^2 - other bigrams
  BIGRAM_N_COUNTS[0] = len(VOCAB)*len(VOCAB) - number_of_bigrams

# computes the new Good Turing counts for all bigrams that appear fewer than
# 5 times in the corpus. Stores the newly computed counts back into the global LM
def compute_good_turing_counts():
  for token, token_dict in LANGUAGE_MODEL.iteritems():
    for second_token, bigram_count in token_dict.iteritems():
      if (second_token != UNIGRAM_COUNT_SYM) and bigram_count < 10:
        # compute the new Good Turing count: c* = (c+1)(N(c+1)/N(c))
        count = LANGUAGE_MODEL[token][second_token]
        LANGUAGE_MODEL[token][second_token] = (count+1) * (BIGRAM_N_COUNTS[count+1] / float(BIGRAM_N_COUNTS[count]))
        # print "before count: {}".format(count)
        # print "n(c+1): {}".format(BIGRAM_N_COUNTS[count+1])
        # print "n(c): {}".format(BIGRAM_N_COUNTS[count])
        # print "turing count: {}".format(LANGUAGE_MODEL[token][second_token])

perplexities = []

def compute_perplexity(file_name, tokens):
  global perplexities
  bigrams = zip(tokens, tokens[1:])
  temp_sum = 0
  for first, second in bigrams:
    if first not in VOCAB:
      first = UNK_WORD
    if second not in LANGUAGE_MODEL[first]:
      second = UNK_WORD

    #print "(first,second): ({}, {})".format(first,second)
    
    following_word_counts = [count for token,count in LANGUAGE_MODEL[first].items() if token != UNIGRAM_COUNT_SYM]
    # known word probability calculation
    bigram_prob = LANGUAGE_MODEL[first][second] / float(sum(following_word_counts))
    # if it was changed to an unknown word, we need to use the UNK count for the probability calculation
    if second == UNK_WORD:
      bigram_prob = LANGUAGE_MODEL[first][UNK_WORD] / float(sum(following_word_counts))
      print "N(1) N(0): {}, {}".format(BIGRAM_N_COUNTS[1], BIGRAM_N_COUNTS[0])
      print "real value: {}".format(float(BIGRAM_N_COUNTS[1]) / BIGRAM_N_COUNTS[0] / sum(following_word_counts))
      print "bigram p: {}".format(bigram_prob)
    # log base 2 used
    temp_sum -= math.log(bigram_prob, 2)
    #print "temp_sum: {}\n".format(temp_sum)

  # include exp and division by length of tokens
  perplexity = math.exp(temp_sum / len(tokens))
  #print "{} perplexity = {}".format(file_name, perplexity)
  perplexities.append(perplexity)

def tokenize_perplexity_file(root):
  for dir_name, sub_dir_list, file_list in os.walk(root):
    if dir_name != TEST_DIR_NAME:
      # selectively filter out hidden files
      file_list = [f for f in file_list if f[0] != '.']
      for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        file_content = open(file_path).read()
        # remove contractions for tokenizing
        file_content = file_content.replace("'", "")
        tokens = nltk.word_tokenize(file_content)
        processed_tokens = preprocess_file_tokens(tokens)
        # compute perplexity of given file tokens
        compute_perplexity(file_name, processed_tokens)





def run():
  if len(sys.argv) < 2:
    print 'Insufficient Arguments: Please provide a corpus to train on!'
  
  # TEST EMAIL
  elif sys.argv[1] == 'test':
    test_sample_string(TEST_EMAIL)
  
  # RANDOM SENTENCE GENERATION
  elif sys.argv[1] == 'random_sentence':
    print "Learning on the {} corpus...\n".format(sys.argv[1])
    tokenize_files(ROOT_DIRECTORY + sys.argv[2])
    words, probabilities = generate_unigram_probability_distribution()
    unigram_sentence = unigram_random_sentence(words, probabilities)
    print "UNIGRAM SENTENCE:\n{}\n".format(unigram_sentence)
    bigram_sentence = bigram_random_sentence()
    print "BIGRAM SENTENCE:\n{}".format(bigram_sentence)
  
  # PERPLEXITY CALCULATION
  elif sys.argv[1] == 'perplexity':
    print "Computing perplexity value for the {} corpus...\n".format(sys.argv[2])
    tokenize_files(ROOT_DIRECTORY + sys.argv[2])
    modify_unknown_words()
    compute_total_bigram_counts()
    compute_good_turing_counts()
    tokenize_perplexity_file(TEST_CLASSIFICATION_DIRECTORY)

    avg_perplexity = sum(perplexities) / float(len(perplexities))
    print "average perplexity: {}".format(avg_perplexity)

  else:
    print 'fill this out'

if __name__=='__main__':
  run()
