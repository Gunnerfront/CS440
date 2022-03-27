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
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # words as keys, dictionary of tags to number of occurrences as values
    words_to_tags = {}
    tag_occurences = {}

    for sentence in train:
        for word_tag_pair in sentence:
            word = word_tag_pair[0]
            tag = word_tag_pair[1]
            if tag == "START" or tag == "END":
                continue
            else:
                if word in words_to_tags.keys():
                    if tag in words_to_tags[word].keys():
                        # increments occurrences of a tag for a seen word
                        words_to_tags[word][tag] += 1
                    else:
                        # add new tag for the seen word
                        words_to_tags[word][tag] = 1
                else:
                    # adds new word to seen words
                    words_to_tags[word] = {tag : 1}
            
            if tag in tag_occurences.keys():
                tag_occurences[tag] += 1
            else:
                tag_occurences[tag] = 1

    # figure out test set tags
    most_common_tag = max(tag_occurences, key=tag_occurences.get)
    tagged_sentences = []
    for sentence in test:
        tagged_sentence = []
        for word in sentence:
            if word == 'START' or word == 'END':
                tagged_sentence.append((word, word))
            else:
                if word in words_to_tags.keys():
                    # give it the most common tag for the seen word
                    words_most_common_tag = max(words_to_tags[word], key=words_to_tags[word].get)
                    tagged_sentence.append((word, words_most_common_tag))
                else:
                    # give it the most common tag of all seen words
                    tagged_sentence.append((word, most_common_tag))
        tagged_sentences.append(tagged_sentence)

    return tagged_sentences

# python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm baseline