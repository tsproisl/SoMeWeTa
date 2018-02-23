#!/usr/bin/env python3

import collections
import json
import math


def read_lexicon(filename):
    """Read in the lexicon."""
    lexicon = collections.defaultdict(set)
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if line == "":
                continue
            word, wc = line.split("\t")
            lexicon[word.lower()].add(wc)
    for k, v in lexicon.items():
        lexicon[k] = list(v)
    return lexicon


def read_brown_clusters(fh):
    """"""
    brown_clusters = {}
    for line in fh:
        cluster, word, freq = line.rstrip().split("\t")
        brown_clusters[word] = (cluster, round(math.log(int(freq))))
    return brown_clusters


def read_mapping(filename):
    """"""
    with open(filename) as fh:
        mapping = json.load(fh)
    return mapping


def read_word2vec_vectors(fh):
    """"""
    word_to_vec = {}
    for line in fh:
        fields = line.rstrip().split()
        word_to_vec[fields[0]] = fields[1:]
    return word_to_vec


def read_word2vec_clusters(fh):
    """"""
    word_to_vec = {}
    for line in fh:
        word, cluster = line.rstrip().split("\t")
        word_to_vec[word] = cluster
    return word_to_vec


def get_sentences(fh, tagged=True):
    """A generator over the sentence in `filename`."""
    sentence = []
    for line in fh:
        line = line.strip()
        if line == "":
            if tagged:
                words, tags = zip(*sentence)
                yield list(words), list(tags)
            else:
                yield sentence
            sentence = []
        else:
            if tagged:
                sentence.append(line.split("\t", 2))
            else:
                sentence.append(line)
    if len(sentence) > 0:
        if tagged:
            words, tags = zip(*sentence)
            yield list(words), list(tags)
        else:
            yield sentence


def read_corpus(fh, tagged=True):
    """Return a list of sentences, each consisting of a list of tokens."""
    words, tags, lengths = [], [], []
    for sentence in get_sentences(fh, tagged):
        if tagged:
            w, t = sentence
            lengths.append(len(w))
            words.extend(w)
            tags.extend(t)
        else:
            lengths.append(len(sentence))
            words.extend(sentence)
    if tagged:
        return words, tags, lengths
    else:
        return words, lengths


def iter_corpus(fh, tagged=True):
    """"""
    words, tags, length = [], [], 0
    for sentence in get_sentences(fh, tagged):
        if tagged:
            words, tags = sentence
            length = len(words)
            yield words, tags, length
        else:
            length = len(sentence)
            words = sentence
            yield words, length


def evaluate(gold, predicted, ignore_tag=None):
    """Evaluate accuracy of predicted against gold."""
    total = len(gold)
    correct = sum(1 for gt, pt in zip(gold, predicted) if gt == pt)
    if ignore_tag is not None:
        total = sum(1 for gt in gold if gt != ignore_tag)
    return correct / total
