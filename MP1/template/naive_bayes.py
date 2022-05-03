# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""

hamLabel = 1
spamLabel = 0
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    #print(len(X),'X')
    pos_vocab = Counter()
    neg_vocab = Counter()
    #TODO:
    for emailNum in range(0, len(X)):
        for word in X[emailNum]:
            if y[emailNum] == hamLabel:
                pos_vocab[word] = pos_vocab[word] + 1
            elif y[emailNum] == spamLabel:
                neg_vocab[word] = neg_vocab[word] + 1
    

    return pos_vocab, neg_vocab


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets
    y: train labels
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab, neg_vocab = create_word_maps_uni(X, y)
    #TODO:
    for emailNum in range(0, len(X)):
        for word in range(0, len(X[emailNum]) - 1):
            bigram = X[emailNum][word] + " " + X[emailNum][word + 1]
            if y[emailNum] == hamLabel:
                pos_vocab[bigram] = pos_vocab[bigram] + 1
            elif y[emailNum] == spamLabel:
                neg_vocab[bigram] = neg_vocab[bigram] + 1
    
    return pos_vocab, neg_vocab



# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")

# Method for finding the total of all values in a Counter object
def counterTotal(counterObject):
    total = 0
    for key, val in counterObject.items():
        total += val
    return total

"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    print("\nInitiating Training Phase...\n")
    hamWordCounts, spamWordCounts = create_word_maps_uni(train_set, train_labels)

    totalHamWordTokens = counterTotal(hamWordCounts)
    totalSpamWordTokens = counterTotal(spamWordCounts)

    print("\nInitiating Dev Phase...\n")
    devLabels = []

    # Fills devLabels with the predicted spam (0) or ham label (1) based on training data word counts using laplace smoothing
    for email in dev_set:
        productOfHamLikelihoods = 0.
        productOfSpamLikelihoods = 0.

        for word in email:
            if word in hamWordCounts:
                pHamWord = (hamWordCounts[word] + laplace) / (totalHamWordTokens + laplace * (1 + len(hamWordCounts)))
                productOfHamLikelihoods = productOfHamLikelihoods + np.log(pHamWord)
            else:
                pHamWord = laplace / (totalHamWordTokens + laplace * (1 + len(hamWordCounts)))
                productOfHamLikelihoods = productOfHamLikelihoods + np.log(pHamWord)
            if word in spamWordCounts:
                pSpamWord = (spamWordCounts[word] + laplace) / (totalSpamWordTokens + laplace * (1 + len(spamWordCounts)))
                productOfSpamLikelihoods = productOfSpamLikelihoods + np.log(pSpamWord)
            else:
                pSpamWord = laplace / (totalSpamWordTokens + laplace * (1 + len(spamWordCounts)))
                productOfSpamLikelihoods = productOfSpamLikelihoods + np.log(pSpamWord)

        pHam = np.log(pos_prior) + productOfHamLikelihoods
        pSpam = np.log(1. - pos_prior) + productOfSpamLikelihoods

        devLabels.append(np.argmax([pSpam, pHam]))          # Appends 0 if probability of spam is higher, 1 if probability of ham is higher
    
    print("\nReturning Dev Labels..\n")
    return devLabels


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None

    print("\nInitiating Training Phase...\n")
    hamWordCounts, spamWordCounts = create_word_maps_uni(train_set, train_labels)
    hamBigramCounts, spamBigramCounts = create_word_maps_bi(train_set, train_labels)

    distinctHamWords = len(hamWordCounts)
    distinctSpamWords = len(spamWordCounts)

    totalHamWordTokens = counterTotal(hamWordCounts)
    totalSpamWordTokens = counterTotal(spamWordCounts)

    totalHamTokens = counterTotal(hamBigramCounts)
    totalSpamTokens = counterTotal(spamBigramCounts)

    distinctHamXValues = len(hamBigramCounts)
    distinctSpamXValues = len(spamBigramCounts)

    print("\nInitiating Dev Phase...\n")
    devLabels = []

    for email in dev_set:
        # Unigram Part: Getting the summations of logs of unigram probabilities to get the unigramProbHam and unigramProbSpam
        uniProductOfHamLikelihoods = 0.
        uniProductOfSpamLikelihoods = 0.
        for word in email:
            if word in hamWordCounts:
                pHamWord = (hamWordCounts[word] + unigram_laplace) / (totalHamWordTokens + unigram_laplace * (1 + distinctHamWords))
                uniProductOfHamLikelihoods = uniProductOfHamLikelihoods + np.log(pHamWord)
            else:
                pHamWord = unigram_laplace / (totalHamWordTokens + unigram_laplace * (1 + distinctHamWords))
                uniProductOfHamLikelihoods = uniProductOfHamLikelihoods + np.log(pHamWord)
            if word in spamWordCounts:
                pSpamWord = (spamWordCounts[word] + unigram_laplace) / (totalSpamWordTokens + unigram_laplace * (1 + distinctSpamWords))
                uniProductOfSpamLikelihoods = uniProductOfSpamLikelihoods + np.log(pSpamWord)
            else:
                pSpamWord = unigram_laplace / (totalSpamWordTokens + unigram_laplace * (1 + distinctSpamWords))
                uniProductOfSpamLikelihoods = uniProductOfSpamLikelihoods + np.log(pSpamWord)
        
        unigramProbHam = np.log(pos_prior) + uniProductOfHamLikelihoods
        unigramProbSpam = np.log(1. - pos_prior) + uniProductOfSpamLikelihoods

        # Bigram Part: Getting the summations of logs of bigram probabilities to get the bigramProbHam and bigramProbSpam
        biProductOfHamLikelihoods = 0.
        biProductOfSpamLikelihoods = 0.
        for word in range(0, len(email) - 1):
            bigram = email[word] + " " + email[word + 1]
            if bigram in hamBigramCounts:
                pHamBigram = (hamBigramCounts[bigram] + bigram_laplace) / (totalHamTokens + bigram_laplace * (1 + distinctHamXValues))
                biProductOfHamLikelihoods = biProductOfHamLikelihoods + np.log(pHamBigram)
            else:
                pHamBigram = bigram_laplace / (totalHamTokens + bigram_laplace * (1 + distinctHamXValues))
                biProductOfHamLikelihoods = biProductOfHamLikelihoods + np.log(pHamBigram)
            if bigram in spamBigramCounts:
                pSpamBigram = (spamBigramCounts[bigram] + bigram_laplace) / (totalSpamTokens + bigram_laplace * (1 + distinctSpamXValues))
                biProductOfSpamLikelihoods = biProductOfSpamLikelihoods + np.log(pSpamBigram)
            else:
                pSpamBigram = bigram_laplace / (totalSpamTokens + bigram_laplace * (1 + distinctSpamXValues))
                biProductOfSpamLikelihoods = biProductOfSpamLikelihoods + np.log(pSpamBigram)
        
        bigramProbHam = np.log(pos_prior) + biProductOfHamLikelihoods
        bigramProbSpam = np.log(1. - pos_prior) + biProductOfSpamLikelihoods
        #print(unigramProbHam, unigramProbSpam)

        # Final Calc
        finalHamProbability = abs(unigramProbHam)**(1 - bigram_lambda) * abs(bigramProbHam)**(bigram_lambda)
        finalSpamProbability = abs(unigramProbSpam)**(1 - bigram_lambda) * abs(bigramProbSpam)**(bigram_lambda)

        decision = np.argmin([finalSpamProbability, finalHamProbability])
        devLabels.append(decision)    # Appends 0 if probability of spam is higher, 1 if probability of ham is higher

    print("\nReturning Dev Labels..\n")
    return devLabels
