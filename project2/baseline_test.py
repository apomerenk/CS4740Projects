import os
import nltk
import glob
import sys
import numpy
import math
from glob import glob
from PyDictionary import PyDictionary

TRAIN_DIR = './train'
TEST_PUBLIC_DIR = './test-public'
TEST_PRIVATE_DIR = './test-private'

def baseline_test():
  print "Baseline test starting...\n"
  uncertain_phrases = []

  for dir_name, sub_dir_list, file_list in os.walk(TRAIN_DIR):
    # ignore hidden files
    file_list = [f for f in file_list if f[0] != '.']

    for file_name in file_list:
      file_path = os.path.join(TRAIN_DIR,file_name)
      file_content = open(file_path)
      uncertain_phrase=[]
      cue_index=0
      
      for line in iter(file_content):
        # don't count empty lines
        if line != '\n':
          line_values = line.split()
          word = line_values[0]
          POS = line_values[1]
          uncertainty = line_values[2]

          if word == '.':
            cue_index=0

          phrase_not_added=True

          if (uncertainty != '_' and int(uncertainty[4:]) > cue_index) or (uncertainty == '_' and phrase_not_added):
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
  
  token_count = -1
  flattened_uncertain_phrases = [val for sublist in uncertain_phrases for val in sublist]
  final_str = "Type,Spans\nCUE-public,"
  new_file = open('kaggle_submission.txt', 'w')
  
  for dir_name, sub_dir_list, file_list in os.walk(TEST_PUBLIC_DIR):
    file_list = [f for f in file_list if f[0] != '.']
    for file_name in file_list:
      file_path = os.path.join(TEST_PUBLIC_DIR,file_name)
      file_content = open(file_path)
      
      for line in iter(file_content):
        # don't count empty lines
        if line != '\n':
          line_values = line.split()
          word = line_values[0]
          POS = line_values[1]

          # record predictions of uncertain words
          if word in flattened_uncertain_phrases:
            final_str += str(token_count) + '-' + str(token_count) + ' '
          
          # count every token encountered
          token_count += 1

  # cut off final space
  final_str = final_str[:-1]
  # add in empty slot for sentence ambiguity prediction
  final_str += '\nCUE-private,'
  token_count = -1

  for dir_name, sub_dir_list, file_list in os.walk(TEST_PRIVATE_DIR):
    file_list = [f for f in file_list if f[0] != '.']
    for file_name in file_list:
      file_path = os.path.join(TEST_PRIVATE_DIR,file_name)
      file_content = open(file_path)
      
      for line in iter(file_content):
        # don't count empty lines
        if line != '\n':
          line_values = line.split()
          word = line_values[0]
          POS = line_values[1]

          # record predictions of uncertain words
          if word in flattened_uncertain_phrases:
            final_str += str(token_count) + '-' + str(token_count) + ' '
          
          # count every token encountered
          token_count += 1

  # cut off final space
  final_str = final_str[:-1]
  new_file.write(final_str)
  new_file.close()
  print token_count

def run():
  baseline_test()

if __name__=='__main__':
  run()
