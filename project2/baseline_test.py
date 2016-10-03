import os
import nltk
import glob
import sys
import numpy
import math
from glob import glob
from PyDictionary import PyDictionary

TRAIN_DIR='./train'

def baseline_test(directory_name):
  print "Baseline test starting...\n"

  uncertain_phrases=[]

  for dir_name, sub_dir_list, file_list in os.walk(directory_name):
    file_list = [f for f in file_list if f[0] != '.']
    index_train=int(len(file_list)*float(.9))
    file_list_train=file_list[:index_train]
    file_list_validation=file_list[index_train:]
    for file_name in file_list_train:
      file_path=os.path.join(directory_name,file_name)
      file_content = open(file_path).read()
      tokens = file_content.split()

      uncertain_phrase=[]
      cue_index=0
      # break into three value elements
      for x in range(0,int(len(tokens)/3)):
        word=tokens[x*3].lower()
        POS=tokens[x*3+1]
        uncertainty=tokens[x*3+2]
        # reached end of sentence
        if word=='.':
          cue_index=0

        phrase_not_added=True

        if (uncertainty!= '_' and int(uncertainty[4:]) > cue_index) or (uncertainty == '_' and phrase_not_added):
          # add phrase to uncertainphrases
          if uncertain_phrase not in uncertain_phrases:
            uncertain_phrases.append(uncertain_phrase)
          # increment cue index
          cue_index+=1
          # reset phrase
          uncertain_phrase=[]
          phrase_not_added=True

        # if the word is an uncertainty
        if uncertainty != '_':
          # check if a new cue index
          # continuation of last uncertain phrase
          uncertain_phrase.append(word)
          phrase_not_added=True

    total_uncertain_tokens = 0
    total_accurate_predictions = 0
    flattened_uncertain_phrases = [val for sublist in uncertain_phrases for val in sublist]

    for file_name in file_list_validation:
      file_path=os.path.join(directory_name,file_name)
      file_content = open(file_path).read()
      tokens = file_content.split()

      # break into three value elements
      for x in range(0,int(len(tokens)/3)):
        word=tokens[x*3].lower()
        POS=tokens[x*3+1]
        uncertainty=tokens[x*3+2]

        # count all the uncertain taggings that appear
        if uncertainty != '_':
          total_uncertain_tokens += 1

        # record accurate predictions of uncertain words
        if word in flattened_uncertain_phrases and uncertainty != '_':
          total_accurate_predictions += 1
  
  print "BASELINE ACCURACY: {} / {} => {}".format(total_accurate_predictions, total_uncertain_tokens, float(total_accurate_predictions)/total_uncertain_tokens)

def run():
  if len(sys.argv) < 2:
    print 'Insufficient Arguments: Please provide valid directory'
  elif sys.argv[1]=='baseline':
    baseline_test(TRAIN_DIR)

if __name__=='__main__':
  run()
