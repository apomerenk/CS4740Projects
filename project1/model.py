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

Also, you will have to install PyDictionary which can quickly be done with PIP:
  > sudo pip install pydictionary

In addition, you will need to have numpy installed in order for us to correctly sample
our probability distributions for each language model.

EXAMPLE USAGES
==============

-Random Sentence Generation:
To generate random sentences from the command line from the root directory labeled 'project1', run:
  > python model.py random_sentence <corpus_to_train_on>
  > python model.py random_sentence autos

-Calculate Perplexity:
  Note: We had to add 1 new line to the beginning of confusion_set.txt so that we didn't parse metadata.
        If you're using your own downloaded version of the data_corrected/ dir, please add the new line. :)
  > python model.py perplexity <unigram/bigram> <corpus_to_train_on>
  > python model.py perplexity unigram space
  > python model.py perplexity bigram religion

-Classification of Test Documents
  > python model.py classification

-Synonym Extension Squeezing:
  > python model.py squeezing <corpus_to_train_on>

-Context Aware Spelling Correction
  > python model.py spelling_correction <corpus_to_train_on>

-Quick test email corpus
  > python model.py test
will run a quick test training on the TEST_EMAIL constant below

If insufficient arguments are supplied to the application, it will prompt for the
relevant command line arguments. 

Enjoy! :)
"""

TEST_ROOT_DIRECTORY = './atheism'
ROOT_DIRECTORY = './data_corrected/classification task/'
SPELLING_ROOT_DIRECTORY = './data_corrected/spell_checking_task/'
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
UNIGRAM_N_COUNTS = dict()
CORPUS_WORD_COUNT = 0
UNK_WORD_COUNT = 0
UNK_WORD_SET = set()
VOCAB = set()
SENTENCE_ENDS = ['?', '!', '.']
UNK_WORD = "unk"
perplexities = []

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

file_name_list=[]
def tokenize_perplexity_file(root, is_bigram = True):
  global file_name_list
  for dir_name, sub_dir_list, file_list in os.walk(root):
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
  # assigned sub_dir_list
  sub_dir_list=['atheism','autos','graphics','medicine','medicine','motorcycles','religion','space']
  for dir_name in sub_dir_list:
    print dir_name
    if dir_name != TEST_DIR_NAME:
      tokenize_files(root+dir_name)
      modify_bigram_unknown_words()
      compute_total_bigram_counts()
      compute_good_turing_bigram_counts()
      tokenize_perplexity_file(TEST_CLASSIFICATION_DIRECTORY, is_bigram=True)

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
    classification=min(all_perplexities[file_name], key = lambda t: t[1])
    all_perplexities[file_name]=classification[0]
    all_perplexities_sorted=sorted(all_perplexities.items(),key= lambda t:int(t[0][5:].rsplit('.',1)[0]))

  for x in all_perplexities_sorted:
    print str(x[0])+","+str(sub_dir_list.index(x[1]))

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

  else:
    print 'Command not understood, please try again.'

if __name__=='__main__':
  run()
