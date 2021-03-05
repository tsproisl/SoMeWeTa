#!/usr/bin/env python3

import collections
import html
import json
import logging
import math
import sys
import time
import xml.etree.ElementTree as ET


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


def get_sentences(fh, tagged=True, warn_threshold=500):
    """A generator over the sentence in `filename`."""
    sentence_counter = 1
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
            sentence_counter += 1
        else:
            if tagged:
                sentence.append(line.split("\t", 2))
            else:
                sentence.append(line)
            if len(sentence) == warn_threshold:
                logging.warn("Sentence %d is extremely long (≥ %d words) – Are you sure that the input sentences are delimited by an empty line?" % (sentence_counter, warn_threshold))
    if len(sentence) > 0:
        if tagged:
            words, tags = zip(*sentence)
            yield list(words), list(tags)
        else:
            yield sentence


def read_corpus(fh, tagged=True):
    """Return a list of sentences, each consisting of a list of tokens."""
    words, tags, lengths = [], [], []
    if tagged:
        for w, t, l in iter_corpus(fh, tagged):
            words.extend(w)
            tags.extend(t)
            lengths.append(l)
        return words, tags, lengths
    else:
        for w, l in iter_corpus(fh, tagged):
            words.extend(w)
            lengths.append(l)
        return words, lengths


def iter_corpus(fh, tagged=True):
    """Yield one sentence at a time, each consisting of a list of
    tokens.

    """
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


def parse_xml(xml, is_file=True):
    """Return a list of XML elements and their text/tail as well as the
    whole text of the document.

    """
    Element = collections.namedtuple("Element", ["element", "type", "text"])

    def text_getter(elem):
        text = elem.text
        tail = elem.tail
        if text is None:
            text = ""
        if tail is None:
            tail = ""
        yield Element(elem, "text", text)
        for child in elem:
            for t in text_getter(child):
                yield t
        yield Element(elem, "tail", tail)
    if is_file:
        tree = ET.parse(xml)
        root = tree.getroot()
    else:
        root = ET.fromstring(xml)
    elements = list(text_getter(root))
    return elements


def read_tagged_xml(xml):
    """Return a list of sentences, each consisting of a list of tokens. If
    tagged=False, also return the original lines and the indexes of
    the words.

    """
    words, tags, lengths = [], [], []
    for w, t, l in iter_xml(xml, tagged=True):
        words.extend(w)
        tags.extend(t)
        lengths.append(l)
    return words, tags, lengths


def iter_xml(xml, tagged=True):
    """Yield one sentence at a time. If tagged=False, also return the
    original lines and the indexes of the words.

    """
    for sentence in get_sentences(xml, tagged=False):
        word_indexes = [i for i, line in enumerate(sentence) if not (line.startswith("<") and line.endswith(">"))]
        if tagged:
            words, tags = zip(*[sentence[i].split("\t", 2) for i in word_indexes])
        else:
            words = [sentence[i] for i in word_indexes]
        words = [html.unescape(w) for w in words]
        length = len(words)
        if tagged:
            yield words, tags, length
        else:
            yield words, length, sentence, word_indexes


def add_pos_to_xml(tagged_sentence, lines, word_indexes):
    """Add part-of-speech tags to original lines of XML file."""
    if len(word_indexes) > 0:
        words, tags = zip(*tagged_sentence)
    else:
        tags = []
    for idx, tag in zip(word_indexes, tags):
        lines[idx] += "\t%s" % tag
    return lines


def int2str(eta):
    """ returns an appropriately formatted version of the number of seconds provided """
    if eta < 2:
        eta = 1000 * eta        # milliseconds
        if eta >= 1:
            return "{:01} ms".format(int(eta))
        else:
            return "<1 ms"

    nr_days = int(eta // (60 * 60 * 24))
    nr_hours = int(eta // (60 * 60) % 24)
    if nr_days > 0:
        return "{:01} days, {:01} hours".format(nr_days, nr_hours)
    else:
        nr_minutes = int(eta // 60 % 60)
        if nr_hours > 12:
            return "{:01} hours, {:01} minutes".format(nr_hours, nr_minutes)
        else:
            nr_seconds = int(eta % 60)
            return "{:02}:{:02}:{:02}".format(nr_hours, nr_minutes, nr_seconds)


class Progress(object):
    """
    Class for showing progress in for-loops
    (1) initialize before loop
    (2) .update every loop
    (3) .finalize after loop
    optional parameters for initialization:
    - length of loop (will calculate approximate ETA)
    - refresh rate (default: every 100 lines)
    """
    def __init__(self, length=None, rate=100):
        self.c = 0
        self.rate = rate
        when = time.time()
        self.start_glob = when
        self.avg_glob = 0
        self.start_rate = when
        self.d = length
        self.eta = 0
        self.last = 0
        self.max_msg_length = 0

    # aliases
    def up(self):
        self.update()

    def fine(self):
        self.finalize()

    # methods
    def update(self, increment=1):
        self.c += increment

        if self.c >= self.last + self.rate:
            when = time.time()
            self.global_speed = self.c / (when - self.start_glob)
            self.current_speed = (self.c - self.last) / (when - self.start_rate)
            self.start_rate = when
            self.last = self.c

            if self.d is not None:
                self.eta = (self.d - self.c) / self.global_speed
                msg = "%3d%% (%d/%d). avg: %5d tokens/s. cur: %5d tokens/s. ETA: %s" % (
                    int(self.c / self.d * 100),
                    self.c,
                    self.d,
                    self.global_speed,
                    self.current_speed,
                    int2str(self.eta)
                )
            else:
                msg = "%d tokens. average: %5d tokens/s. current: %5d tokens/s." % (
                    self.c,
                    self.global_speed,
                    self.current_speed,
                )
            msg_length = len(msg)
            if msg_length < 79:
                msg = " " * (79 - msg_length) + msg
            msg_length = len(msg)
            if msg_length > self.max_msg_length:
                self.max_msg_length = msg_length
            trail = " " * (self.max_msg_length - msg_length)
            print(msg + trail, end="\r", file=sys.stderr)

        if self.c == self.d:
            self.finalize()

    def finalize(self):
        print(" " * self.max_msg_length, end="\r", file=sys.stderr)
