# Gain-Based Feature Selection

> John Lewis

> Portland State University - CS441

> HW3

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

The results were ~70% accuracy for the naive bayes with 10-way cross-validation and ~80-85% accuracy for the id3 with 10-way cross-validation and 0.01 gain limit

## Extra Credit
### Changing the number of features:

#### Naive Bayes Learner Performance by Number of Features

|5  |10 |20 |50 |100    |200    |300    |600    |
|---|---|---|---|-------|-------|-------|-------|
|acc:0.803 fpr:0.085 fnr:0.112 |acc:0.833 fpr:0.060 fnr:0.108 |acc:0.800 fpr:0.067 fnr:0.133 |acc:0.842 fpr:0.080 fnr:0.078 |acc:0.817 fpr:0.092 fnr:0.092 |acc:0.828 fpr:0.062 fnr:0.110 |acc:0.796 fpr:0.080 fnr:0.124    |acc:0.814 fpr:0.096  fnr:0.089 |
|acc:0.750 fpr:0.122 fnr:0.128 |acc:0.844 fpr:0.050 fnr:0.106 |acc:0.807 fpr:0.067 fnr:0.126 |acc:0.819 fpr:0.078 fnr:0.103 |acc:0.821 fpr:0.085 fnr:0.094 |acc:0.810 fpr:0.108 fnr:0.083 |acc:0.798 fpr:0.099 fnr:0.103    |acc:0.849 fpr:0.069  fnr:0.083 |
|acc:0.782 fpr:0.108 fnr:0.110 |acc:0.830 fpr:0.067 fnr:0.103 |acc:0.817 fpr:0.076 fnr:0.108 |acc:0.839 fpr:0.055 fnr:0.106 |acc:0.844 fpr:0.076 fnr:0.080 |acc:0.828 fpr:0.080 fnr:0.092 |acc:0.846 fpr:0.078 fnr:0.076    |acc:0.844 fpr:0.069  fnr:0.087 |
|acc:0.773 fpr:0.106 fnr:0.122 |acc:0.835 fpr:0.055 fnr:0.110 |acc:0.823 fpr:0.050 fnr:0.126 |acc:0.830 fpr:0.048 fnr:0.122 |acc:0.821 fpr:0.078 fnr:0.101 |acc:0.826 fpr:0.085 fnr:0.089 |acc:0.856 fpr:0.069 fnr:0.076    |acc:0.833 fpr:0.071  fnr:0.096 |
|acc:0.814 fpr:0.057 fnr:0.128 |acc:0.826 fpr:0.050 fnr:0.124 |acc:0.846 fpr:0.053 fnr:0.101 |acc:0.819 fpr:0.087 fnr:0.094 |acc:0.823 fpr:0.080 fnr:0.096 |acc:0.826 fpr:0.078 fnr:0.096 |acc:0.844 fpr:0.071 fnr:0.085    |acc:0.821 fpr:0.089  fnr:0.089 |
|acc:0.775 fpr:0.108 fnr:0.117 |acc:0.839 fpr:0.055 fnr:0.106 |acc:0.860 fpr:0.064 fnr:0.076 |acc:0.835 fpr:0.060 fnr:0.106 |acc:0.842 fpr:0.073 fnr:0.085 |acc:0.810 fpr:0.094 fnr:0.096 |acc:0.844 fpr:0.085 fnr:0.071    |acc:0.805 fpr:0.119  fnr:0.076 |
|acc:0.778 fpr:0.124 fnr:0.099 |acc:0.858 fpr:0.034 fnr:0.108 |acc:0.837 fpr:0.046 fnr:0.117 |acc:0.812 fpr:0.080 fnr:0.108 |acc:0.837 fpr:0.064 fnr:0.099 |acc:0.846 fpr:0.076 fnr:0.078 |acc:0.823 fpr:0.092 fnr:0.085    |acc:0.833 fpr:0.080  fnr:0.087 |
|acc:0.805 fpr:0.087 fnr:0.108 |acc:0.807 fpr:0.067 fnr:0.126 |acc:0.837 fpr:0.057 fnr:0.106 |acc:0.798 fpr:0.071 fnr:0.131 |acc:0.819 fpr:0.089 fnr:0.092 |acc:0.867 fpr:0.053 fnr:0.080 |acc:0.819 fpr:0.085 fnr:0.096    |acc:0.814 fpr:0.092  fnr:0.094 |
|acc:0.755 fpr:0.126 fnr:0.119 |acc:0.817 fpr:0.067 fnr:0.117 |acc:0.839 fpr:0.055 fnr:0.106 |acc:0.823 fpr:0.064 fnr:0.112 |acc:0.837 fpr:0.089 fnr:0.073 |acc:0.862 fpr:0.053 fnr:0.085 |acc:0.821 fpr:0.073 fnr:0.106    |acc:0.844 fpr:0.083  fnr:0.073 |
|acc:0.798 fpr:0.088 fnr:0.114 |acc:0.828 fpr:0.049 fnr:0.123 |acc:0.849 fpr:0.060 fnr:0.091 |acc:0.800 fpr:0.100 fnr:0.100 |acc:0.828 fpr:0.074 fnr:0.098 |acc:0.821 fpr:0.107 fnr:0.072 |acc:0.828 fpr:0.067 fnr:0.105    |acc:0.853 fpr:0.077  fnr:0.070 |
|**AVG:</br>acc:0.783 fpr:0.101 fnr:0.116</br></br>ACC VAR:</br>0.064** | **AVG: </br>acc:0.832 fpr:0.055 fnr:0.113</br></br>ACC VAR:</br>0.051** |**AVG: </br>acc:0.832 fpr:0.06 fnr:0.109</br></br>ACC VAR:</br> 0.060**    |**AVG: </br>acc:0.822 fpr:0.0723 fnr:0.106</br></br>ACC Var:</br>0.044**  |**AVG: </br>acc:0.829 fpr:0.080 fnr:0.091</br></br>ACC VAR:</br>0.027**    |**AVG:</br>acc:0.832 fpr:0.080 fnr:0.088</br></br>ACC VAR:</br>0.057**  |**AVG:</br>acc:0.828 fpr:0.080 fnr:0.093</br></br>ACC VAR: 0.060**  |**AVG:</br>acc:0.831 fpr:0.085 fnr:0.084</br></br>ACC VAR: 0.048** |

#### id3 Learner Performance by Number of Features

|5  |10 |20 |50 |100    |200    |300    |1000    |
|---|---|---|---|-------|-------|-------|-------|
|acc:0.796 fpr:0.087 fnr:0.117 |acc:0.780 fpr:0.030 fnr:0.190   |acc:0.821 fpr:0.014 fnr:0.165  |acc:0.739 fpr:0.005 fnr:0.257  |acc:0.780 fpr:0.000 fnr:0.220  |acc:0.743 fpr:0.000 fnr:0.257  |acc:0.631 fpr:0.002 fnr:0.367  |acc:0.665 fpr:0.000 fnr:0.335 |
|acc:0.803 fpr:0.085 fnr:0.112 |acc:0.819 fpr:0.069 fnr:0.112   |acc:0.784 fpr:0.023 fnr:0.193  |acc:0.789 fpr:0.005 fnr:0.206  |acc:0.764 fpr:0.000 fnr:0.236  |acc:0.679 fpr:0.000 fnr:0.321  |acc:0.720 fpr:0.000 fnr:0.280  |acc:0.631 fpr:0.000 fnr:0.369 |
|acc:0.784 fpr:0.094 fnr:0.122 |acc:0.812 fpr:0.025 fnr:0.163   |acc:0.805 fpr:0.011 fnr:0.183  |acc:0.796 fpr:0.002 fnr:0.202  |acc:0.761 fpr:0.000 fnr:0.239  |acc:0.759 fpr:0.000 fnr:0.241  |acc:0.679 fpr:0.000 fnr:0.321  |acc:0.688 fpr:0.000 fnr:0.312 |
|acc:0.784 fpr:0.099 fnr:0.117 |acc:0.828 fpr:0.057 fnr:0.115   |acc:0.796 fpr:0.030 fnr:0.174  |acc:0.800 fpr:0.005 fnr:0.195  |acc:0.752 fpr:0.002 fnr:0.245  |acc:0.782 fpr:0.000 fnr:0.218  |acc:0.766 fpr:0.000 fnr:0.234  |acc:0.638 fpr:0.000 fnr:0.362 |
|acc:0.773 fpr:0.115 fnr:0.112 |acc:0.835 fpr:0.032 fnr:0.133   |acc:0.796 fpr:0.009 fnr:0.195  |acc:0.764 fpr:0.002 fnr:0.234  |acc:0.771 fpr:0.002 fnr:0.227  |acc:0.695 fpr:0.000 fnr:0.305  |acc:0.713 fpr:0.000 fnr:0.287  |acc:0.647 fpr:0.000 fnr:0.353 |
|acc:0.794 fpr:0.101 fnr:0.106 |acc:0.814 fpr:0.046 fnr:0.140   |acc:0.794 fpr:0.016 fnr:0.190  |acc:0.771 fpr:0.011 fnr:0.218  |acc:0.693 fpr:0.000 fnr:0.307  |acc:0.745 fpr:0.000 fnr:0.255  |acc:0.755 fpr:0.000 fnr:0.245  |acc:0.720 fpr:0.000 fnr:0.280 |
|acc:0.761 fpr:0.108 fnr:0.131 |acc:0.819 fpr:0.037 fnr:0.144   |acc:0.787 fpr:0.021 fnr:0.193  |acc:0.750 fpr:0.005 fnr:0.245  |acc:0.761 fpr:0.005 fnr:0.234  |acc:0.748 fpr:0.000 fnr:0.252  |acc:0.757 fpr:0.000 fnr:0.243  |acc:0.674 fpr:0.000 fnr:0.326 |
|acc:0.778 fpr:0.122 fnr:0.101 |acc:0.771 fpr:0.030 fnr:0.200   |acc:0.778 fpr:0.030 fnr:0.193  |acc:0.766 fpr:0.002 fnr:0.232  |acc:0.778 fpr:0.000 fnr:0.222  |acc:0.771 fpr:0.000 fnr:0.229  |acc:0.766 fpr:0.000 fnr:0.234  |acc:0.679 fpr:0.000 fnr:0.321 |
|acc:0.766 fpr:0.106 fnr:0.128 |acc:0.830 fpr:0.062 fnr:0.108   |acc:0.810 fpr:0.009 fnr:0.181  |acc:0.814 fpr:0.005 fnr:0.181  |acc:0.794 fpr:0.000 fnr:0.206  |acc:0.739 fpr:0.002 fnr:0.259  |acc:0.757 fpr:0.000 fnr:0.243  |acc:0.690 fpr:0.000 fnr:0.310 |
|acc:0.793 fpr:0.095 fnr:0.112 |acc:0.821 fpr:0.021 fnr:0.158   |acc:0.802 fpr:0.016 fnr:0.181  |acc:0.728 fpr:0.002 fnr:0.270  |acc:0.730 fpr:0.000 fnr:0.270  |acc:0.742 fpr:0.000 fnr:0.258  |acc:0.723 fpr:0.000 fnr:0.277  |acc:0.693 fpr:0.000 fnr:0.307 |

#### Conclusion of Feature Numbers

The id3 decision tree learner accuracy doesn't change drastically with the number of features between 5 and 600. There is a general trend toward false positives and false negatives of around 0.08. The false positives stay relatively unchanged while the false negatives decrease from around 0.01. There is also a slight trend toward higher accuracy with lower variance with the increase in features. It appears that precision rather than accuracy is the biggest change as features increase. Of course, this has diminishing returns and the run time grows in magnitude as the features are increased for the learner.

The Naive Bayes would seem to have something strange happening. The accuracy decreases with the number of features from ~80% with only 5 features to ~70% with 1000 features. At 100 features, the learner begins to show 0 for false positives and increasingly larger false negatives. I wonder if there is something going on that is meant to help the spam/ham filtering that skews toward false negatives rather than false positives that is not translating well to our classes. In the spam/ham instance, it is desirable to avoid false positives that might stop valid emails while accepting false negatives and spam in your inbox now and then isn't as disruptive as missing an important email.





























