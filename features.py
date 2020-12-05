#!/usr/bin/env python3

import sys
import numpy as np
import math
import string



N = 300   # Maximum features after feature selection

# File locations of novels this program uses for the dataset
NOVELS = ["./hw-authors/austen-northanger-abbey.txt",
          "./hw-authors/shelley-frankenstein.txt",
          "./hw-authors/austen-pride-and-prejudice.txt",
          "./hw-authors/shelley-the-last-man.txt"]




class Dataset:
  
  pgID = 0                # Unique ID incremented for each paragraph added
  dictionary = dict()     # Key = All unique words in dataset, Value = set(pgIDs)
  authorPgs = {0: set(),  # Key = AuthorID, Value = set(pgIds)
               1: set()}
  headers = ["pgID",      # Headers for binary matrix features
             "AuthID"]     
  binary = []             # Binary features of paragraphs, None is placeholder

  # List of characters to remove from provided text
  to_remove = string.punctuation.replace("'","") + "—“”‘’0123456789"



  # Clears dataset fields and resets for new data
  def reset(self):
    self.paragraphs = dict()
    self.pgID = 0


 
  # Takes a string
  # Returns 1 if string contains "shelley"
  # Returns 0 if string contains "austen"
  # Returns -1 if neither substring exists
  def idAuthor(self, text):
    if("austen" in text):
      return 0
    elif("shelley" in text):
      return 1
    else:
      return -1



  # Reads text and adds unique words and paragraph ID as key, value pair to self.dictionary
  # Adds author ID (0|1) and paragraph ID as key, value pair to self.authorPgs
  # Increments self.pgID for each paragraph read
  def read(self, text):

    with open(text, "r") as toRead:

      for line in toRead:

        # If line is end of paragraph, start new paragraph word list
        if(line == "\n"):

          # Increment unique paragraph ID
          self.pgID += 1
          continue
        
        # Add paragraph ID to author set
        self.authorPgs[self.idAuthor(text)].add(self.pgID)

        # Remove all punctuation from line except apostrophes
        line = line.replace("--"," ")
        line = line.translate(str.maketrans("", "", self.to_remove)).strip('\n')

        # Split line into words
        for word in line.split(' '):

          # Remove any remaining apostrophes that aren't internal
          if(word.startswith("'")):
            word = word[1:]
          if(word.endswith("'")):
            word = word[:-1]

          word = word.lower()
          # Add non single-character words to dictionary if word with authorID, pgID tuple
          if(len(word) > 1):
            if word not in self.dictionary:
              self.dictionary[word] = {self.pgID}
            else:
              self.dictionary[word].add(self.pgID)
    self.pgID += 1



  # Builds a binary categorical encoding for each word found in dataset
  # Adds to self.headers for each word in dataset to track what columns represent
  def buildBinary(self):

    # Start with a matrix of zeros with rows = paragraphs, columns = authorID + words
    self.binary = np.zeros((self.pgID, len(self.dictionary)+2), dtype=int)

    self.binary[:,0] = np.arange(self.pgID)

    # Set author ID column to 1 for Shelley, leave 0 for Austen
    self.binary[list(self.authorPgs[1]),1] = 1

    # Set index to 1 if paragraph contains word based on self.dictionary
    for i, word in enumerate(self.dictionary, start=2):

      # keep track of which columns represent which words (pandas-ish headers for np)
      self.headers.append(word)
     
      # Get the set of pgIDs as a list and set those indices to 1 in matrix
      idx = list(self.dictionary[word])
      self.binary[idx, i] = 1

  
  
  # Calculate the entropy from probabilities of classA and classB
  # Returns a float of the calculated entropy
  def calcEntropy(self, classA, classS):

    if(classA == 0):
      entropy = -1 * (classS * math.log(classS, 2))
    elif(classS == 0):
      entropy = -1 * (classA * math.log(classA, 2))
    else:
      entropy = -1 * (classS * math.log(classS, 2) + classA * math.log(classA, 2))
    return entropy



  # Calculates final entropy from splitting dataset on a word based on probabilities
  # of each class and entropy calculated from the split
  # Returns the gain as a float
  def calcFinalEnt(self, word):

    # Total paragraphs where word occurs
    postotal = len(self.dictionary[word])

    # Find the number of Shelley/Austen classes have word
    austen = len(self.dictionary[word].intersection(self.authorPgs[0]))
    shelley = len(self.dictionary[word].intersection(self.authorPgs[1]))

    # Probability of Shelley/Austen based on word
    classS = shelley/postotal  
    classA = 1 - classS     

    # Calculate entropy for having word
    hasEntropy = self.calcEntropy(classA, classS)

    # Total paragraphs word does not occur
    negtotal = (self.pgID) - postotal
    

    # Probability of Shelley/Austen based on not having word
    classS = (len(self.authorPgs[1]) - shelley)/negtotal
    classA = 1 - classS

    # Calculate entropy for not having word
    hasntEntropy = self.calcEntropy(classA, classS)

    # Adjusted entropy given probability of being 1 or 0 for that feature
    return ((postotal/self.pgID) * hasEntropy + (negtotal/self.pgID) * hasntEntropy)



  # Tracks the gain calculated for each split and selects the top N features
  # Updates self.binary with the newly selected N features
  def gainSelection(self):

    # Hold the gain of each word
    wordGain = dict()

    # How many total paragraphs have been read
    total = self.pgID
   
    # Starting entropy based on number of Shelley and Austen instances
    pShelley = np.sum(self.binary[:,1])/total
    pAusten = 1 - pShelley
    startEntropy = -1 * (pShelley * math.log(pShelley, 2) + pAusten * math.log(pAusten, 2))

    # Calculate gain for splitting on each word
    for i in range(2,len(self.dictionary)):
      finalEntropy = self.calcFinalEnt(self.headers[i])
      wordGain[i] = startEntropy - finalEntropy

    # Get the word indexes of the top N features
    sortedKeys = sorted(wordGain, key=wordGain.get)
    n = (-1 * N) - 1
    top = sortedKeys[-1:n:-1]
    top.extend([0,1])
    top.sort()

    # update headers to match new feature list 
    self.headers = [self.headers[i] for i in top]

    # Update binary feature matrix
    self.binary = self.binary[:,top]




def main():
  
  data = Dataset()

  for novel in NOVELS:
    data.read(novel)
  
  data.buildBinary()

  data.gainSelection()

  np.savetxt(sys.stdout, data.binary, delimiter=",", fmt="%i")
  return 0




if __name__ == "__main__":
  main()

