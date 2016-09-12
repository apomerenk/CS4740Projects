import os
import nltk
import glob
import sys
import numpy

TEST_ROOT_DIRECTORY = './atheism'
ROOT_DIRECTORY = './data_corrected/classification task/'
TEST_DIR_NAME = 'test_for_classification'
PUNCT_TO_REMOVE = ['(', ')', '<', '>', ',', '/', '-', '"', 
                   ':', '``', '*', '[', ']', '#', '|', "'", '$', '%',
                   '^', '~', "`", "\\", '_', '+', '{', '}']
UNIGRAM_COUNT_SYM = '*'
LANGUAGE_MODEL = dict()
CORPUS_WORD_COUNT = 0
SENTENCE_ENDS = ['?', '!', '.']

TEST_STRING = """"From : lpzsml@unicorn.nott.ac.uk ( Steve Lang ) Subject : Re : Objective Values ' v ' Scientific Accuracy ( was Re : After 2000 years , can we say that Christian Morality is ) In article <C5J718.Jzv@dcs.ed.ac.uk> , tk@dcs.ed.ac.uk ( Tommy Kelly ) wrote : > In article <1qjahh$mrs@horus.ap.mchp.sni.de> frank@D012S658.uucp ( Frank O'Dwyer ) writes : > > > Science ( " the real world " ) has its basis in values , not the other way round , > > as you would wish it . > > You must be using ' values ' to mean something different from the way I > see it used normally . > > And you are certainly using ' Science ' like that if you equate it to > " the real world " . > > Science is the recognition of patterns in our perceptions of the Universe > and the making of qualitative and quantitative predictions concerning > those perceptions . Science is the process of modeling the real world based on commonly agreed interpretations of our observations ( perceptions ) . > It has nothing to do with values as far as I can see . > Values are ... well they are what I value . > They are what I would have rather than not have - what I would experience > rather than not , and so on . Values can also refer to meaning . For example in computer science the value of 1 is TRUE , and 0 is FALSE . Science is based on commonly agreed values ( interpretation of observations ) , although science can result in a reinterpretation of these values . > Objective values are a set of values which the proposer believes are > applicable to everyone . The values underlaying science are not objective since they have never been fully agreed , and the change with time . The values of Newtonian physic are certainly different to those of Quantum Mechanics . Steve Lang SLANG -> SLING -> SLINK -> SLICK -> SLACK -> SHACK -> SHANK -> THANK -> THINK -> THICK"""

def test_sample_string(test):
  tokens = nltk.word_tokenize(test)
  processed_tokens = preprocess_file_tokens(tokens)
  update_language_model(processed_tokens)
  print LANGUAGE_MODEL

def update_language_model(tokens):
  global CORPUS_WORD_COUNT
  for i in range(len(tokens)-1):
    # keep track of total words in the corpus
    CORPUS_WORD_COUNT += 1
    # make dict keys consistent
    token = tokens[i].lower()
    next_token = tokens[i+1].lower()
    #print "token: {}".format(token)
    #print "next_token: {}".format(next_token)
    if token in LANGUAGE_MODEL:
      # increment unigram count of string
      LANGUAGE_MODEL[token][UNIGRAM_COUNT_SYM] += 1
      # increment bigram count of 2-element sequence
      if next_token in LANGUAGE_MODEL[token]:
        LANGUAGE_MODEL[token][next_token] += 1
      else:
        LANGUAGE_MODEL[token][next_token] = 1
    else:
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
  else:
    LANGUAGE_MODEL[final_token] = dict()
    LANGUAGE_MODEL[final_token][UNIGRAM_COUNT_SYM] = 1

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
  print "DIR: {}".format(os.getcwd())
  for dir_name, sub_dir_list, file_list in os.walk(root):
    print "DIRECTORY: {}\n".format(dir_name)
    
    if dir_name != TEST_DIR_NAME:
      # selectively filter out hidden files
      file_list = [f for f in file_list if f[0] != '.']
      for file_name in file_list:
        #print "FILE: {}\n".format(file_name)
        file_path = os.path.join(dir_name, file_name)
        file_content = open(file_path).read()
        # remove contractions for tokenizing
        file_content = file_content.replace("'", "")
        tokens = nltk.word_tokenize(file_content)
        processed_tokens = preprocess_file_tokens(tokens)
        #print processed_tokens
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

def run():
  if sys.argv[1] == 'test':
    test_sample_string(TEST_STRING)
  elif len(sys.argv) < 2:
    print 'Please provide a corpus to train on!'
  else:
    print "Learning on {} corpus".format(sys.argv[1])
    tokenize_files(ROOT_DIRECTORY + sys.argv[1])
    words, probabilities = generate_unigram_probability_distribution()
    
    for i in range(10):
      unigram_sentence = unigram_random_sentence(words, probabilities)
      print unigram_sentence

    print '\n\n\n'

    for i in range(10):
      bigram_sentence = bigram_random_sentence()
      print bigram_sentence

    #print LANGUAGE_MODEL

if __name__=='__main__':
  run()
