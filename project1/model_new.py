import os
import nltk
import glob

ROOT_DIRECTORY = './data_corrected/classification task'
TEST_DIR_NAME = 'test_for_classification'
PUNCT_TO_REMOVE = ['(', ')', '<', '>', ',', '/', '-', '"', ':', '``', '*', '[', ']']

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

def tokenize_files(root = ROOT_DIRECTORY):
  print "DIR: {}".format(os.getcwd())
  for dir_name, sub_dir_list, file_list in os.walk(root):
    print "DIRECTORY: {}\n".format(dir_name)
    
    if dir_name != TEST_DIR_NAME:
      # selectively filter out hidden files
      file_list = [f for f in file_list if f[0] != '.']
      for file_name in file_list:
        print "FILE: {}\n".format(file_name)
        file_path = os.path.join(dir_name, file_name)
        file_content = open(file_path).read()
        tokens = nltk.word_tokenize(file_content)
        
        processed_tokens = preprocess_file_tokens(tokens)
        print processed_tokens

def run():
  tokenize_files()

if __name__=='__main__':
  run()
