import os
import nltk
import glob
import sys
import numpy
import math
from glob import glob
from PyDictionary import PyDictionary



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
TEST_DIR_NAMEFULL='./data_corrected/classification task/test_for_classification/'
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

# used for our synonym squeezing
DICTIONARY=PyDictionary()

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

def generate_synonyms(token):
  return []


map_dictionary=dict()
def squeeze_language_model(threshold):

# unigram probability distribution is same squeezed vs unsqueezed since it just looks at language model
# unigram sentences are the same, just impossible to be a low probability word
  global map_dictionary
  # get all words that were in the unk set
  # check if the count is less than or equal to threshhold
  for token,token_dict in LANGUAGE_MODEL.items():
    # check if token falls below threshhold and isnt a sentence marker
    if token_dict[UNIGRAM_COUNT_SYM]<=threshold and token not in SENTENCE_ENDS:
      # generate synonyms
      raw_synonyms=(DICTIONARY.synonym(token))
      # if there are no synonyms, add the empty list
      synonyms=[]
      if raw_synonyms!=None:
        # synonyms=raw_synonyms
        for x in raw_synonyms:
          try:synonyms.append(str(x))
          except:synonyms.append(UNK_WORD)
      # format the synonyms


      # get corresponding words to add to new found word in language model
      second_words=[]
      for second_word,count in token_dict.items():
        if second_word != UNIGRAM_COUNT_SYM:
          second_words.append((second_word,count))
      # iterate through ranked synonyms
      counter=0
      replaced=False
      i=0
      # go through synonyms while stopping when it gets replaced
      while(len(synonyms)>0 and not replaced and i<len(synonyms)):
      # for i in range(0,len(synonyms)):
        # if the word is in the model, add the word into the language model for the synonym
        if synonyms[i] in LANGUAGE_MODEL:
          replaced=True
          # increment the synonym count
          LANGUAGE_MODEL[synonyms[i]][UNIGRAM_COUNT_SYM]+=token_dict[UNIGRAM_COUNT_SYM]
          # add all the word's next words to the synonym, and update their count
          for z,count in second_words:
            if z in LANGUAGE_MODEL[synonyms[i]]:
              LANGUAGE_MODEL[synonyms[i]][z]+=count
            else:
              LANGUAGE_MODEL[synonyms[i]][z]=count
          # add the original word to the map dictionary to keep the functionality of synonyms
          map_dictionary[token]=synonyms[i]

          # remove the old word from the language model
          del LANGUAGE_MODEL[token]
      # TODO: if count is 1, remove from unk words
        i+=1


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

def bigram_squeezed_sentence():
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
    # change in squeezed vs unsqueezed
    if last_word not in LANGUAGE_MODEL:
      last_word=map_dictionary[last_word]
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
    # ##########
    # ONLY THING THAT CHANGES IN SQUEEZED VS UNSQUEEZED
    #############
    if next_word not in LANGUAGE_MODEL:
      next_word=map_dictionary[next_word]
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
      # print "N(1) N(0): {}, {}".format(BIGRAM_N_COUNTS[1], BIGRAM_N_COUNTS[0])
      # print "real value: {}".format(float(BIGRAM_N_COUNTS[1]) / BIGRAM_N_COUNTS[0] / sum(following_word_counts))
      # print "bigram p: {}".format(bigram_prob)
    # log base 2 used
    temp_sum -= math.log(bigram_prob, 2)
    #print "temp_sum: {}\n".format(temp_sum)

  # include exp and division by length of tokens
  perplexity = math.exp(temp_sum / len(tokens))
  #print "{} perplexity = {}".format(file_name, perplexity)
  perplexities.append(perplexity)

file_name_list=[]
def tokenize_perplexity_file(root):
  global file_name_list
  for dir_name, sub_dir_list, file_list in os.walk(root):
    # print file_name_list
    if dir_name != TEST_DIR_NAME:
      # selectively filter out hidden files
      file_list = [f for f in file_list if f[0] != '.']
      file_name_list=file_list
      for file_name in file_list:
        file_path = os.path.join(dir_name, file_name)
        file_content = open(file_path).read()
        # remove contractions for tokenizing
        file_content = file_content.replace("'", "")
        tokens = nltk.word_tokenize(file_content)
        processed_tokens = preprocess_file_tokens(tokens)
        # compute perplexity of given file tokens
        compute_perplexity(file_name, processed_tokens)

all_perplexities=dict()

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]

# call on ROOT_DIRECTORY
def classified_files(root):
  global LANGUAGE_MODEL
  global perplexities
  global BIGRAM_N_COUNTS
  global CORPUS_WORD_COUNT
  global UNK_WORD_COUNT
  global UNK_WORD_SET
  global VOCAB
  global file_name_list
  sub_list=[]
  sub_dir_list=get_immediate_subdirectories(root)
  for dir_name in sub_dir_list:
    print dir_name
    if dir_name != TEST_DIR_NAME:
      tokenize_files(root+dir_name)
      modify_unknown_words()
      compute_total_bigram_counts()
      compute_good_turing_counts()
      tokenize_perplexity_file(TEST_CLASSIFICATION_DIRECTORY)

      for x in range(0,len(perplexities)):
        file_name=file_name_list[x]
        if file_name in all_perplexities:
          all_perplexities[file_name].append((dir_name,perplexities[x]))
        else:
          all_perplexities[file_name]=[]
          all_perplexities[file_name].append((dir_name,perplexities[x]))

      print 'perplexities calculated'
      LANGUAGE_MODEL=dict()
      BIGRAM_N_COUNTS = dict()
      CORPUS_WORD_COUNT = 0
      UNK_WORD_COUNT = 0
      UNK_WORD_SET = set()
      VOCAB = set()
      perplexities=[]
  print 'all perplexities calculated, calculating minimums'

  # go through all_perplexities and find lowest
  for file_name in all_perplexities:
    # print file_name
    # file_perplexities=all_perplexities[file_name]
    # lowest_index=file_perplexities.index(min(file_perplexities))
    classification=min(all_perplexities[file_name], key = lambda t: t[1])
    # classification=sub_dir_list[lowest_index]
    all_perplexities[file_name]=classification[0]
    all_perplexities_sorted=sorted(all_perplexities.items(),key= lambda t:int(t[0][5:].rsplit('.',1)[0]))

  for x in all_perplexities_sorted:
    print "File name: ",x[0],"; classification: ",x[1]




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

  # CLASSIFICATION
  elif sys.argv[1] == 'classification':
    print "classifying"
    classified_files(ROOT_DIRECTORY)

  # SQUEEZING
  elif sys.argv[1] == 'squeezing':
    print "SQUEEZING on the {} corpus...\n".format(sys.argv[1])
    print "Heads up, this will take a while to go through all the API calls to get synonyms (ETA 10-20 minutes)"
    print "ignore the warning below"
    tokenize_files(ROOT_DIRECTORY + sys.argv[2])
    squeeze_language_model(2)
    words, probabilities = generate_unigram_probability_distribution()
    for i in range(0,50):
      bigram_sentence = bigram_squeezed_sentence()
      print "BIGRAM SENTENCE:\n{}".format(bigram_sentence)

    # avg_perplexity = sum(perplexities) / float(len(perplexities))
    # print "average perplexity: {}".format(avg_perplexity)

  else:
    print 'fill this out'

if __name__=='__main__':
  run()
