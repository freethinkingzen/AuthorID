# Gain-Based Feature Selection

John Lewis
Portland State University
CS441
HW3a

## Usage

`python3 features.py >[output file]`

## Processing the texts

This takes text files from <http://github.com/pdx-cs-ai/hw-authors> that have had the front and back, proper names, and other content removed
leaving only the text of the novels.

Those files are then read one paragraph at a time by **features.py** doing the following:

    1. Assigning and storing a paragraphID to an authorID [0|1] to associate paragraphs with authors
    2. Reading each line
    3. Removing all numbers and punctuation except apostrophes
    4. Splitting the lines into words
    5. Checking for any remaining punctuation
    6. Ignoring 1 letter words
    7. Convert to lowercase
    8. Storing each word as a key for a dictionary structure and the paragraph ID as a value
 
 The decision to leave in internal punctuation, change the words to lowercase, and filter words by length are probably worth playing with. I wanted to keep the internal punctuation to keep the meanings of things like "its" vs "it's" and "were" vs "we're" becuase the use of contractions might indicate a more casual author (not likely given the time period but a hypothesis). The word lenth was implemented by recommendation in the assignment and might be worth changing to see what results that will yield. I know that "one" had popped into the top 300, not becuase there was a difference in the number of times it was used by an author, but becuase of the number of times it was not used. I did remove the lowercase which resulted in a larger total dictionary but did not change the accuracy of the machine learners so I left it in to speed up the program when I was struggling with lists and 2D list instead of dictionaries and sets.
    
## Creating a binary feature set

A numpy matrix of zeros is built to hold `x-axis = num paragraph` and `y-axis = paragraphID + authorID + num words` and the paragraphIDs and authorIDs are copied to the matrix

For each word column in the matrix, the dictionary looks up all the paragraphsIDs that contain that word and set those indices to 1, leaving the ones that do not as 0

The result is a matrix of ~4000 paragraphs and ~17000 words:

    pgID0, AuthID, Feature1, Feature2, Feature3 ...
    pgID1, AuthID, Feature1, Feature2, Feature3 ...
    pgID2, AuthID, Feature1, Featyue2, Feature3 ...
    ...


## Calculating Gain to filter features

In order to limit the number of features, gain-based feature selection is used to select the top *N* features

First the probability of the novel being written by Jane Austen or Mary Shelly is calculated:

    P(Shelley) = Paragraphs by Shelly / Total Paragraphs
    P(Austen) = Paragraphs by Austen / Total Paragraphs

These are used to calculate the starting entropy of the data:

    U_start = -(P(Austen) * log_2(P(Austen)) + P(Shelley) * log_2(P(Shelley))

Next, the data is split on each word so that the paragraphs that have the word are in one group and the paragraphs that do not are in another group.

Instead of actually splitting the groups which would require memory and time, the program uses the dictionary to count the number of paragraphs that are stored for that word. This gets us the number of positive instances. Subtracting from that number from the total paragraphs gives us the number of negative instances.
 
    PosTotal = len(dictionary[word])
    NegTotal = total paragraphs - PosTotal
    
To find the number of times Austen and Shelley use a word, `intersection` is used on the set of paragraphs for a word and the set of paragraphs for an author and the length of that will be the number of times a paragraph contains the word for that author.

    AustenUses = len(dictionary[word].intersection(author[0]))
    ShelleyUses = len(dictionary[word].intersection(author[1]))
    
The probabilities of Austen and Shelley using the word in a paragraph are calculated. Since there are only two classes, we can simply subtract the result of the first one calculated from 1 to find the second:

    P(A+) = AustenUses / PosTotal
    P(S+) = 1 - p(A+)
    
The entropy of the word being used is calculated:

    U_+ = -(P(A+) * log_2(P(A+)) + P(S+) * log_2(P(S+)))
    
To find the number of times Austen and Shelly *DON't* use a word, we simply subtract `AustenUses` and `ShelleyUses` from the total numberthe probabilities of Austen and Shelley *NOT* using the word are calculated:

    P(A-) = (len(author[0]) - AustenUses) / NegTotal
    P(S-) = 1 - P(A-)
    
The entropy of the word *NOT* being used by an author is calculated:

    U_- = -(P(A-) * log_2(P(A-)) + P(S-) * log_2(P(S-)))
    
Then the final entropy of the word can be calculated by adjusting for the probability of pos or neg:

    U_final = (PosTotal/total paragraphs) * U_+ + (NegTotal/total paragraphs) * U_-
    
The gain for that feature is:

    Gain = U_start - U_final
    
This gain is added to a dictionary that holds the paragraphID as a key and the gain as a value. Once the gain is calculated for every word, the dictionary is sorted by value and the keys are extracted as a list. This list is used to only keep those idices of the binary feature matrix. So if features 10, 36, and 782 were the top N features. The matrix would now contain:

    pgID1, AuthID, Feature10, Feature36, Feature782
    pgID2, AuthID, Feature10, Feature36, Feature782
    pgID3, AuthID, Feature10, Feature36, Feature782
    ...
    
## Machine Learner

For this assignment I used the provided psam learner from <http://github.com/pdx-cs-ai/psamlearn> trying both the naive bayes and id3 learners.

The results were ~70% accuracy for the naive bayes with 10-way cross-validation and ~80% accuracy for the id3 with 10-way cross-validation and 0.01 gain limit
    
    

    
