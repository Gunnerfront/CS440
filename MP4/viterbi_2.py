# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import math
import numpy as np

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    laplace_k = 0.00001
    word_tag_pair_occurrences = {}
    tag_pair_occurences = {}
    tag_occurences = {}
    num_tags = 0
    num_sentences = 0
    tags = []

    # get data from training set
    for sentence in train:
        prev_tag = 'START'
        for word_tag_pair in sentence:
            word = word_tag_pair[0]
            tag = word_tag_pair[1]

            if not tag in tags:
                tags.append(tag)
                num_tags += 1

            if tag in tag_occurences.keys():
                tag_occurences[tag] += 1
            else:
                tag_occurences[tag] = 1

            if word in word_tag_pair_occurrences.keys():
                if tag in word_tag_pair_occurrences[word].keys():
                    # increments occurrences of a tag for a seen word
                    word_tag_pair_occurrences[word][tag] += 1
                else:
                    # add new tag for the seen word
                    word_tag_pair_occurrences[word][tag] = 1
            else:
                word_tag_pair_occurrences[word] = {tag : 1}

            tag_pair = (prev_tag, tag)
            if tag_pair in tag_pair_occurences.keys():
                tag_pair_occurences[tag_pair] += 1
            else:
                tag_pair_occurences[tag_pair] = 1

            prev_tag = tag
        
        num_sentences += 1
    
    # get hapax set for bettering accuracy of unseen words
    hapax = {}
    for word in word_tag_pair_occurrences.keys():
        if len(word_tag_pair_occurrences[word].keys()) == 1 and list(word_tag_pair_occurrences[word].values())[0] == 1:
            tag = list(word_tag_pair_occurrences[word].keys())[0]
            hapax[tag] = hapax.get(tag, 0) + 1
    num_hapax_words = sum(hapax.values())

    # get initial tag probability distribution, pi
    pi = []
    for tag in tags:
        occur = tag_pair_occurences.get(('START', tag), 0.)
        prob = math.log((occur + laplace_k) / (num_sentences + laplace_k * num_tags))
        pi.append(prob)

    # Construct transition probabilities
    a = []
    for tag_i in tags:
        tag_i_to_j = []
        for tag_j in tags:
            this_a = math.log((tag_pair_occurences.get((tag_i, tag_j), 0.) + laplace_k) / (tag_occurences[tag_i] + laplace_k * num_tags))
            tag_i_to_j.append(this_a)
        a.append(tag_i_to_j)
    
    # Construct emission probabilities inside word_tag dict
    V = len(word_tag_pair_occurrences.keys())       # number of distinct words in training set
    for word in word_tag_pair_occurrences.keys():
        for tag in word_tag_pair_occurrences[word].keys():
            word_tag_pair_occurrences[word][tag] = math.log((word_tag_pair_occurrences[word][tag] + laplace_k) / (tag_occurences[tag] + laplace_k * (V + 1.)))

    # figure out test set tags
    tagged_sentences = []
    for sentence in test:
        tagged_sentence = []
        trellis = []
        word_count = 0
        
        for word in sentence:
            # Constructing the trellis
            prob_tag_pairs = []
            if word_count == 0:
                # if at start of sentence
                for tag in tags:
                    tag_idx = get_tag_index_2(tags, tag)
                    log_b = 0.
                    if word in word_tag_pair_occurrences.keys() and tag in word_tag_pair_occurrences[word].keys():
                        log_b = word_tag_pair_occurrences[word].get(tag, 0)
                    elif word in word_tag_pair_occurrences.keys():
                        log_b = math.log(laplace_k / (tag_occurences[tag] + laplace_k * (V + 1)))
                    else:
                        log_b =  math.log((hapax.get(tag, 0) + laplace_k) / (num_hapax_words + laplace_k * (V + 1)))
                    prob_tag_pair = (log_b + pi[tag_idx], tag)      # (v_j1, tag)
                    prob_tag_pairs.append(prob_tag_pair)
            else:
                for tag in tags:  # these represent the current tag nodes
                    node_probs_for_this_tag = []
                    outer_tag_idx = get_tag_index_2(tags, tag)
                    for inner_tag in tags:  # these represent the previous tag nodes
                        inner_tag_idx = get_tag_index_2(tags, inner_tag)
                        prev_node_prob_for_this_tag = trellis[word_count - 1][inner_tag_idx][0]
                        log_b = 0.
                        if word in word_tag_pair_occurrences.keys() and tag in word_tag_pair_occurrences[word].keys():
                            log_b = word_tag_pair_occurrences[word].get(tag, 0)
                        elif word in word_tag_pair_occurrences.keys():
                            log_b = math.log(laplace_k / (tag_occurences[tag] + laplace_k * (V + 1)))
                        else:
                            log_b = math.log((hapax.get(tag, 0) + laplace_k) / (num_hapax_words + laplace_k * (V + 1)))
                        e = a[inner_tag_idx][outer_tag_idx] + log_b
                        v = prev_node_prob_for_this_tag + e
                        node_probs_for_this_tag.append(v)

                    prob_tag_pairs.append( (max(node_probs_for_this_tag), tags[np.argmax(node_probs_for_this_tag)]) )

            word_count += 1
            trellis.append(prob_tag_pairs)
        
        # Make tagged sentence based on backtracking from trellis
        tagged_sentence = TrellisToTaggedSentence_2(trellis, sentence)
        tagged_sentences.append(tagged_sentence)

    return tagged_sentences

def get_tag_index_2(tags, tag):
    return tags.index(tag)

def TrellisToTaggedSentence_2(trellis, sentence):
    tagged_sentence = [('END', 'END')]
    trellix_idx = len(trellis) - 1
    sentence = sentence[0:-1]
    while trellix_idx >= 1:
        max_node_prob = float('-inf')
        max_tag = ''
        curr_v_jts = trellis[trellix_idx]
        for prob_tag_pair in curr_v_jts:
            if prob_tag_pair[0] > max_node_prob:
                max_tag = prob_tag_pair[1]
                max_node_prob = prob_tag_pair[0]
        tagged_sentence.append( (sentence[trellix_idx - 1], max_tag) )
        trellix_idx -= 1
    
    tagged_sentence.reverse()
    return tagged_sentence

# python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_2