import os
import glob

corpus="the students liked the assignment"
corpussplit=str.split(corpus)

def p(word1, word2 = ""):
  global corpussplit
  a=corpussplit
  if (word2 == ""):
    # one argument
    prob = a.count(word1)/ float(len(a))
    return prob
  else:
    # two arguments
    newcount = 0
    # get the number of times word1 appears
    oldcount = a.count(word1)
    # start breaking down the string
    while a != []:
      # base case where word1 isnt there
      if not word1 in a:
        break
      # get the index of word1
      firstindex=a.index(word1)
      # if its at the start of the string, word2 cant have been before it
      # so go to next loop with new string
      if firstindex == 0:
        a = a[firstindex+1 :]
        continue
      # if the word before word1 is word2, increment counter
      if a[firstindex-1] == word2:
        newcount+=1
        print "hello"
      # modify string to be substring starting just after word1
      a = a[firstindex+1 :]
    return newcount/float(oldcount)

# TEST CASES

# print "p(the)"
# print p(corpussplit[0])
# print "p(liked)"
# print p(corpussplit[2])
# print "P(the|liked)"
# print p(corpussplit[0], corpussplit[2])
# print "P(students|the)"
# print p(corpussplit[1], corpussplit[0])


# Generates all the words in the overall directory
def generate_all_words():
  all_words = dict()
  print os.getcwd()
  # go through all the directories and update the dictionary
  # for filename in os.listdir('/Users/alexpomerenk/Documents/F16/CS4740/Projects/project1/data_corrected/classification task'):
  #   print filename
  # print os.listdir('/Users/alexpomerenk/Documents/F16/CS4740/Projects/project1/data_corrected/classification task')
  # print os.listdir(os.getcwd()+'/data_corrected/classification task')
  
  words = []
  for filename in os.listdir(os.getcwd()+'/data_corrected/classification task'):
    # print filename
    if not filename =='test_for_classification':
      # print filename
      words = generate_directory_words(os.getcwd()+'/data_corrected/classification task/'+filename+'/train_docs', all_words)
      all_words = words
  return words

# Generate the words in one directory
def generate_directory_words(directory, currentwords):
  # go through all the files
  print directory
  for filename in glob.glob(os.path.join(directory,'*.txt')):
    print filename
    openfile = (open(filename)).read()
    # split the words in te fiel
    all_words = openfile.split()
    for word in all_words:
      # add word to dictionary if not there
      if not word in current_words:
        current_words.append(word)
  return current_words


words = generate_all_words()
# print words

