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
SPELLING_ROOT_DIRECTORY = './data_corrected/spell_checking_task/'
TEST_CLASSIFICATION_DIRECTORY = './data_corrected/classification task/test_for_classification'
TEST_DIR_NAME = 'test_for_classification'
PUNCT_TO_REMOVE = ['(', ')', '<', '>', ',', '/', '-', '"', 
                   ':', '``', '*', '[', ']', '#', '|', "'", '$', '%',
                   '^', '~', "`", "\\", '_', '+', '{', '}', '=']
UNIGRAM_COUNT_SYM = '*'
LANGUAGE_MODEL = dict()
# mapping from (<bigram appearance count>, <total number of bigrams with that count>)
BIGRAM_N_COUNTS = dict()
UNIGRAM_N_COUNTS = dict()
CORPUS_WORD_COUNT = 0
UNK_WORD_COUNT = 0
UNK_WORD_SET = set()
VOCAB = set()
SENTENCE_ENDS = ['?', '!', '.']
UNK_WORD = "unk"
perplexities = []

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

  if final_token in LANGUAGE_MODEL:
    LANGUAGE_MODEL[final_token][UNIGRAM_COUNT_SYM] += 1
    # subtract out 1-time appearing words from the VOCAB set
    VOCAB -= UNK_WORD_SET
    VOCAB.add(UNK_WORD)
  else:
    # new word, so record it as an <unk> to start
    UNK_WORD_COUNT += 1
    UNK_WORD_SET.add(final_token)
    LANGUAGE_MODEL[final_token] = dict()
    LANGUAGE_MODEL[final_token][UNIGRAM_COUNT_SYM] = 1
    # subtract out 1-time appearing words from the VOCAB set
    VOCAB -= UNK_WORD_SET
    VOCAB.add(UNK_WORD)

def modify_bigram_unknown_words():
  # aggregate outer unk words into one key
  LANGUAGE_MODEL[UNK_WORD] = dict()
  for token, token_dict in LANGUAGE_MODEL.items():
    # if in set, it only appeared once in the corpus
    if token in UNK_WORD_SET:
      del LANGUAGE_MODEL[token]
      for second_token, bigram_count in token_dict.items():
        # delete the token from the LM as it was only seen 1x and is an UNK
        if second_token != UNIGRAM_COUNT_SYM:
          if second_token in LANGUAGE_MODEL[UNK_WORD]:
            LANGUAGE_MODEL[UNK_WORD][second_token] += bigram_count
          else:
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

def modify_unigram_unknown_words():
  LANGUAGE_MODEL[UNK_WORD] = dict()
  LANGUAGE_MODEL[UNK_WORD][UNIGRAM_COUNT_SYM] = 0
  for token, token_dict in LANGUAGE_MODEL.items():
    # count all <= 1 occurence unigrams as UNK words
    unigram_count = LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM]
    if unigram_count <= 1:
      LANGUAGE_MODEL[UNK_WORD][UNIGRAM_COUNT_SYM] += unigram_count
      del LANGUAGE_MODEL[token]
    
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

def compute_total_unigram_counts():
  number_of_unigrams = 0
  for token, token_dict in LANGUAGE_MODEL.iteritems():
    unigram_count = LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM]
    if unigram_count > 0:
      number_of_unigrams += 1
    if unigram_count <= 5:
      if unigram_count in UNIGRAM_N_COUNTS:
        UNIGRAM_N_COUNTS[unigram_count] += 1
      else:
        UNIGRAM_N_COUNTS[unigram_count] = 1
  
  # we do not compute the UNIGRAM_N_COUNTS[0] value as it does not exist
  # single appearance unigrams are simply the UNK word count
  UNIGRAM_N_COUNTS[1] = LANGUAGE_MODEL[UNK_WORD][UNIGRAM_COUNT_SYM]
  print UNIGRAM_N_COUNTS

# computes the new Good Turing counts for all bigrams that appear fewer than
# 5 times in the corpus. Stores the newly computed counts back into the global LM
def compute_good_turing_bigram_counts():
  for token, token_dict in LANGUAGE_MODEL.iteritems():
    for second_token, bigram_count in token_dict.iteritems():
      if second_token != UNIGRAM_COUNT_SYM and bigram_count < 10:
        # compute the new Good Turing count: c* = (c+1)(N(c+1)/N(c))
        count = LANGUAGE_MODEL[token][second_token]
        LANGUAGE_MODEL[token][second_token] = (count+1) * (BIGRAM_N_COUNTS[count+1] / float(BIGRAM_N_COUNTS[count]))

def compute_good_turing_unigram_counts():
  for token, token_dict in LANGUAGE_MODEL.iteritems():
    count = LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM]
    if count >= 1 and count <= 4:
      LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM] = (count+1) * (UNIGRAM_N_COUNTS[count+1] / float(UNIGRAM_N_COUNTS[count]))
      #print "{} => {}".format(count, LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM])

def compute_bigram_perplexity(file_name, tokens):
  global perplexities
  bigrams = zip(tokens, tokens[1:])
  temp_sum = 0
  for first, second in bigrams:
    if first not in VOCAB:
      first = UNK_WORD
    if second not in LANGUAGE_MODEL[first]:
      second = UNK_WORD
    following_word_counts = [count for token,count in LANGUAGE_MODEL[first].items() if token != UNIGRAM_COUNT_SYM]
    bigram_prob = LANGUAGE_MODEL[first][second] / float(sum(following_word_counts))
    temp_sum -= math.log(bigram_prob)

  # include exp and division by length of tokens
  perplexity = math.exp(temp_sum / len(tokens))
  perplexities.append(perplexity)

def compute_unigram_perplexity(file_name, tokens):
  global perplexities
  temp_sum = 0
  count = 0
  for unigram in tokens:
    if unigram not in VOCAB:
      count += 1
      unigram = UNK_WORD
    unigram_prob = LANGUAGE_MODEL[unigram][UNIGRAM_COUNT_SYM] / float(CORPUS_WORD_COUNT)
    temp_sum -= math.log(unigram_prob)

  perplexity = math.exp(temp_sum / len(tokens))
  perplexities.append(perplexity)

def tokenize_perplexity_file(root, is_bigram = False):
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
        if is_bigram:
          compute_bigram_perplexity(file_name, processed_tokens)
        else:
          compute_unigram_perplexity(file_name, processed_tokens)

def gather_confusion_set(root = SPELLING_ROOT_DIRECTORY):
  file_path = os.path.join(root, 'confusion_set.txt')
  file_content = open(file_path)
  misspelled_words = dict()
  first_loop = True
  for line in file_content:
    # skip the first metadata line
    if first_loop:
      first_loop = False
      continue
    split = line.split()
    misspelled_words[split[0]] = split[1]
    misspelled_words[split[1]] = split[0]
  return misspelled_words

def tokenize_spelling_test_files(root, misspelled_words):
  final_string = ""
  # train the model on relevant training docs for the corpus
  for dir_name, sub_dir_list, file_list in os.walk(root+'/train_docs'):
    file_list = [f for f in file_list if f[0] != '.']
    for file_name in file_list:
      file_path = os.path.join(dir_name, file_name)
      file_content = open(file_path).read()
      tokens = file_content.split()
      update_language_model(tokens)
      
  # tokenize new test files for correction (without any pre-processing)
  for dir_name, sub_dir_list, file_list in os.walk(root+'/train_modified_docs'):
    file_list = [f for f in file_list if f[0] != '.']
    for file_name in file_list:
      file_path = os.path.join(dir_name, file_name)
      file_content = open(file_path).read()
      tokens = file_content.split()

      corrected_content = correct_spelling_mistakes(tokens, misspelled_words)
      new_file = open(root+ '/test_docs/' + file_name.rsplit('.')[0] + '_corrected.txt', 'w')
      for token in corrected_content:
        final_string += token + ' '
      new_file.write(final_string[:-1])
      new_file.close()
      final_string = ""

def retreive_higher_probability(first, second, misspelled_words):
  opt_1 = second
  # grab the other possible spelling for comparison
  opt_2 = misspelled_words[second]

  if opt_1 in LANGUAGE_MODEL[first.lower()]:
    p_1 = LANGUAGE_MODEL[first.lower()][opt_1]
  else:
    p_1 = 0
  if opt_2 in LANGUAGE_MODEL[first.lower()]:
    p_2 = LANGUAGE_MODEL[first.lower()][opt_2]
  else:
    p_2 = 0

  if p_1 > p_2:
    return opt_1
  elif p_1 < p_2:
    return opt_2
  else:
    # if they are equal (or both 0), we don't make any action
    # we should only modify the corpus if we know for sure we ought to
    return second

def correct_spelling_mistakes(tokens, misspelled_words):
  bigrams = zip(tokens, tokens[1:])
  adjusted_tokens = []
  # always add the first word
  adjusted_tokens.append(tokens[0])
  for first, second in bigrams:
    if second in misspelled_words and first.lower() in LANGUAGE_MODEL:
      correct_word = retreive_higher_probability(first, second, misspelled_words)
      adjusted_tokens.append(correct_word)
    else:
      adjusted_tokens.append(second)
  return adjusted_tokens

# -------------------------------------------------------------------------------------------------
#                              COMMAND LINE PARSING AND HIGH-LEVEL LOGIC
# -------------------------------------------------------------------------------------------------
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
    print "Computing the {} perplexity value for the {} corpus...\n".format(sys.argv[2], sys.argv[3])
    if sys.argv[2] == 'unigram':
      tokenize_files(ROOT_DIRECTORY + sys.argv[3])
      modify_unigram_unknown_words()
      compute_total_unigram_counts()
      compute_good_turing_unigram_counts()
      tokenize_perplexity_file(TEST_CLASSIFICATION_DIRECTORY, is_bigram = False)
      avg_perplexity = sum(perplexities) / float(len(perplexities))
      print "Average Unigram Perplexity ({}): {}".format(sys.argv[3], avg_perplexity)

    if sys.argv[2] == 'bigram':
      tokenize_files(ROOT_DIRECTORY + sys.argv[3])
      modify_bigram_unknown_words()
      compute_total_bigram_counts()
      compute_good_turing_bigram_counts()
      tokenize_perplexity_file(TEST_CLASSIFICATION_DIRECTORY, is_bigram = True)
      avg_perplexity = sum(perplexities) / float(len(perplexities))
      print "Average Bigram Perplexity ({}): {}".format(sys.argv[3], avg_perplexity)

  # CONTEXT AWARE SPELLING CORRECTION
  elif sys.argv[1] == 'spelling_correction':
    print "Training for spelling correction on the {} corpus...\n".format(sys.argv[2])
    root = SPELLING_ROOT_DIRECTORY + sys.argv[2]
    misspelled_words = gather_confusion_set()
    print "Correcting spelling errors using the {} corpus language model...\n".format(sys.argv[2])
    tokenize_spelling_test_files(root, misspelled_words)
    print "Spelling corrected! Find the corrected files here:\n{}/data_corrected/spell_checking_task/{}/test_docs".format(os.getcwd(), sys.argv[2])

  else:
    print 'fill this out'

if __name__=='__main__':
  run()
