# Python script for setting up NLP env for CS 4740
import os

print 'Downloading Natural Language Toolkit...'
os.system('sudo pip install -U nltk')

print 'Attempting to import nltk...'
import nltk &&

print 'Setup finished!'
