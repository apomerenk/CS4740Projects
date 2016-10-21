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
    # test_list = ['test.txt']

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

def classify_tokens_and_sentences(directory,adverb_ext,threshold):
  EMISSION_COUNTS = Counter(WORD_BIO_LIST)
  _, bio_list = zip(*WORD_BIO_LIST)
  TOTAL_BIO_COUNTS = Counter(bio_list)
  token_bio_classifications = []


  final_str=""
  for dir_name, sub_dir_list, file_list in os.walk(directory):
    # ignore hidden files
    file_list = [f for f in file_list if f[0] != '.']
    start_index=-1
    current_index=0
    uncertain=False

    for file_name in file_list:
      file_path = os.path.join(directory,file_name)
      file_content = open(file_path)
      last_bio = 'O'
      for line in iter(file_content):
        if line != '\n':
          line_values = line.split()
          word = line_values[0]
          # adverb modifier
          POS= line_values[1]
          adverbs=['RB','RBR','RBS']
          adverb_modifier=1
          if (POS in adverbs) and adverb_ext:
            adverb_modifier=1.3

          # compute P(word | tag) * P(this_tag | last_tag) for each bio tag
          computed_bio_products = []
          for tag in ['B', 'I', 'O']:
            p_word_given_tag = EMISSION_COUNTS[(word, tag)] / float(TOTAL_BIO_COUNTS[tag])
            p_transition = TRANSITION_DATA[last_bio][tag] / float(TOTAL_BIO_COUNTS[last_bio])
            product = p_word_given_tag*p_transition*adverb_modifier
            computed_bio_products.append((product, tag))
          max_prob_tag = max(computed_bio_products, key=lambda x:x[0])
          token_bio_classifications.append(max_prob_tag[1])
          last_bio=max_prob_tag[1]
          # new uncertain sequence
          if last_bio=='B':
            if uncertain:
              print start_index, current_index
              # add the last uncertain sequence to string
              final_str+= str(start_index)+'-'+str(current_index-1)+' '
            # update start index
            start_index=current_index
            uncertain=True
          elif last_bio=='O' and uncertain:
            final_str+= str(start_index)+'-'+str(current_index-1)+' '
            uncertain=False
          current_index+=1
        else:
          # it is an empty line, so mark this in the growing list of tags
          # with a sentence delimiter '*'
          token_bio_classifications.append('*')
          # current_index+=1
    print current_index
    # print token_bio_classifications

    sentence_string=""
    condensed=''.join(token_bio_classifications)
    sentence_array=condensed.split('*')
    for x in range(0,len(sentence_array)):
      # threshold modifier
      uncertaintycount=sentence_array[x].count('B')+sentence_array[x].count('I')
      try:
        UCprob=uncertaintycount/float(len(sentence_array[x]))
      except:
        UCprob=0
      if (UCprob>=.1) and threshold:
        sentence_string+=str(x)+' '
      # without threshold
      else if sentence_array[x].find('B')!=-1:
        sentence_string+=str(x)+' '

    return final_str,sentence_string


def generate_kaggle(adverb, threshold):
  words_header = "Type,Spans\nCUE-public,"
  sentences_header = "Type,Indices\nSENTENCE-public,"
  words_middle='\nCUE-private,'
  sentences_middle='\nSENTENCE-private,'
  # classify the public folder, getting sentence and word kaggle format
  classify_public=classify_tokens_and_sentences(TEST_PUBLIC_DIR,adverb,threshold)
  public_words=classify_public[0][:-1]
  public_sentences=classify_public[1][:-1]
  # classify the private folder, getting sentence and word kaggle format
  classify_private=classify_tokens_and_sentences(TEST_PRIVATE_DIR,adverb,threshold)
  private_words=classify_private[0][:-1]
  private_sentences=classify_private[1][:-1]
  # build strings for kaggle words and sentences
  final_words=words_header+public_words+words_middle+private_words[:-1]
  final_sentences=sentences_header+public_sentences+sentences_middle+private_sentences[:-1]
  # write words to file
  new_file = open('kaggle_words.txt', 'w')
  new_file.write(final_words)
  new_file.close()
  # write sentences to file
  new_file = open('kaggle_sentences.txt', 'w')
  new_file.write(final_sentences)
  new_file.close()


def run():
  generate_probs()

  if len(sys.argv) < 3:
    print 'Insufficient Arguments: Please provide booleans for extensions\n arg1: adverb \n arg2: threshold'
  else:
    generate_kaggle(sys.argv[1],sys.argv[2])
if __name__=='__main__':
  run()
