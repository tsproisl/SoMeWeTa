#!/usr/bin/env python3

import collections
import itertools
import json
import logging
import math
import re
import warnings
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


def read_xml(xml, tagged=True, is_file=True):
    elements = parse_xml(xml, is_file)
    whole_text = "".join((e.text for e in elements))
    whole_text = re.sub(r"^\n+", "", whole_text)
    whole_text = re.sub(r"\n+$", "\n", whole_text)
    whole_text = re.sub(r"\n{3,}", "\n\n", whole_text)
    return read_corpus(whole_text.split("\n"), tagged), elements


def recreate_xml(tagged_tokens, elements):
    """Introduce the tags from tagged_tokens into the elements and create
    string with XML.

    """
    agenda = list(reversed(tagged_tokens))
    for element in elements:
        original_tokens = element.text.split("\n")
        output = []
        for ot in original_tokens:
            if ot == "":
                output.append(ot)
            elif len(agenda) > 0 and agenda[-1][0] == ot:
                tt = agenda.pop()
                output.append("\t".join(tt))
            else:
                warnings.warn("Cannot match tagged tokens with XML input.")
        if len(output) > 0:
            tagged_text = "\n".join(output)
        else:
            tagged_text = "\n"
            warnings.warn("Output should not be empty.")
        if element.type == "text":
            element.element.text = tagged_text
        elif element.type == "tail":
            element.element.tail = tagged_text
    try:
        assert len(agenda) == 0
    except AssertionError:
        warnings.warn("AssertionError: %d tokens left over" % len(agenda))
    xml = ET.tostring(elements[0].element, encoding="unicode")
    if xml[-1] != "\n":
        xml += "\n"
    return xml
