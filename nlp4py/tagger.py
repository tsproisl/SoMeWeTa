#!/usr/bin/env python3

import functools
import gzip
import itertools
import json
import logging
import math
import multiprocessing
import re
import statistics

from nlp4py.averaged_structured_perceptron import AveragedStructuredPerceptron


class ASPTagger(AveragedStructuredPerceptron):
    """A part-of-speech tagger based on the averaged structured
    perceptron.

    """
    def __init__(self, beam_size, iterations, lexicon=None, mapping=None, brown_clusters=None, word_to_vec=None, prior_weights=None):
        super().__init__(beam_size=beam_size, iterations=iterations, latent_features=None, prior_weights=prior_weights)
        self.lexicon = lexicon
        self.mapping = mapping
        self.brown_clusters = brown_clusters
        self.word_to_vec = word_to_vec
        if self.mapping is not None:
            self.mapping["<START-2>"] = "<START>"
            self.mapping["<START-1>"] = "<START>"

    def train(self, words, tags, lengths):
        """"""
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in words])
        X = self._get_static_features(words, lengths)
        self.fit(X, tags, lengths)

    def tag(self, words, lengths):
        """"""
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in words])
        X = self._get_static_features(words, lengths)
        tags = super().predict(X, lengths)
        start = 0
        for length, local_tags in zip(lengths, tags):
            local_words = words[start:start + length]
            start += length
            yield zip(local_words, local_tags)

    def evaluate(self, words, tags, lengths):
        """"""
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in words])
        X = self._get_static_features(words, lengths)
        accuracy = super().score(X, tags, lengths)
        return accuracy

    def crossvalidate(self, words, tags, lengths):
        """"""
        sentence_ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        X = self._get_static_features(words, lengths)
        div, mod = divmod(len(sentence_ranges), 10)
        cvi = functools.partial(self._cross_val_iteration,
                                words=words, X=X, y=tags,
                                lengths=lengths,
                                sentence_ranges=sentence_ranges,
                                div=div, mod=mod)
        with multiprocessing.Pool() as pool:
            accuracies = pool.map(cvi, range(10))
        # accuracies = list(map(cvi, range(10)))
        return statistics.mean(accuracies), 2 * statistics.stdev(accuracies)

    def save(self, filename):
        """"""
        with gzip.open(filename, 'wb') as f:
            # f.write(json.dumps((self.lexicon, self.brown_clusters, self.word_to_vec, self.weights), ensure_ascii=False, indent=4).encode())
            f.write(json.dumps((self.lexicon, self.mapping, self.brown_clusters, self.word_to_vec, self.weights), ensure_ascii=False, indent=4).encode())

    def load(self, filename):
        """"""
        with gzip.open(filename, 'rb') as f:
            model = json.loads(f.read().decode())
            self.lexicon, self.mapping, self.brown_clusters, self.word_to_vec, self.weights = model

    def _cross_val_iteration(self, i, words, X, y, lengths, sentence_ranges, div, mod):
        """"""
        test_ranges = sentence_ranges[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)]
        test_start = test_ranges[0][0]
        test_end = test_ranges[-1][0] + test_ranges[-1][1]
        test_lengths = lengths[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)]
        train_lengths = lengths[:i * div + min(i, mod)] + lengths[(i + 1) * div + min(i + 1, mod):]
        test_words = words[test_start:test_end]
        test_X = X[test_start:test_end]
        test_y = y[test_start:test_end]
        train_words = words[:test_start] + words[test_end:]
        train_X = X[:test_start] + X[test_end:]
        train_y = y[:test_start] + y[test_end:]
        # train
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in train_words])
        self.fit(train_X, train_y, train_lengths)
        # evaluate
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in test_words])
        accuracy = self.score(test_X, test_y, test_lengths)
        logging.info("Accuracy: %.2f%%" % (accuracy * 100,))
        return accuracy

    def _get_static_features(self, words, lengths):
        """"""
        lexicon = self.lexicon
        brown_clusters = self.brown_clusters
        word_to_vec = self.word_to_vec
        features = []
        start = 0
        for length in lengths:
            sentence = words[start:start + length]
            start += length
            tokens = ["<START-2>", "<START-1>"] + [w.lower() for w in sentence] + ["<END+1>", "<END+2>"]
            for i, word in enumerate(sentence):
                j = i + 2
                local_features = set()
                w = tokens[j]
                p1 = tokens[j - 1]
                p2 = tokens[j - 2]
                n1 = tokens[j + 1]
                n2 = tokens[j + 2]
                # constant bias feature acts like a prior
                local_features.add("bias")
                # rounded logarithm of word length
                local_features.add("W_loglength: %d" % round(math.log(len(word))))
                # current word
                local_features.add("W_word: %s" % w)
                # next words
                local_features.add("N1_word: %s" % n1)
                local_features.add("N2_word: %s" % n2)
                # affixes
                local_features.add("W_prefix: %s" % w[:3])
                local_features.add("W_suffix: %s" % w[-3:])
                if i >= 1:
                    local_features.add("P1_suffix: %s" % p1[-3:])
                if length - i > 1:
                    local_features.add("N1_suffix: %s" % n1[-3:])
                # word shape
                local_features.add("W_shape: %s" % self._word_shape(word))
                # Flags
                if i >= 2:
                    local_features.update(self._word_flags(p2, "P2"))
                if i >= 1:
                    local_features.update(self._word_flags(p1, "P1"))
                local_features.update(self._word_flags(w, "W"))
                if length - i > 1:
                    local_features.update(self._word_flags(n1, "N1"))
                if length - i > 2:
                    local_features.update(self._word_flags(n2, "N2"))
                # Brown clusters
                if brown_clusters is not None:
                    # P2, P1, W, N1, N2
                    if i >= 2:
                        bc, freq = brown_clusters.get(p2, ("N/A", 0))
                        local_features.add("P2_brown: %s" % bc)
                    if i >= 1:
                        bc, freq = brown_clusters.get(p1, ("N/A", 0))
                        local_features.add("P1_brown: %s" % bc)
                    bc, freq = brown_clusters.get(w, ("N/A", 0))
                    local_features.add("W_brown: %s" % bc)
                    local_features.add("W_logfreq: %d" % freq)
                    if length - i > 1:
                        bc, freq = brown_clusters.get(n1, ("N/A", 0))
                        local_features.add("N1_brown: %s" % bc)
                    if length - i > 2:
                        bc, freq = brown_clusters.get(n2, ("N/A", 0))
                        local_features.add("N2_brown: %s" % bc)
                if word_to_vec is not None:
                    # if w in word_to_vec:
                    #     for i, d in enumerate(word_to_vec[w]):
                    #         local_features.add("W_w2v_%d: %d" % (i, round(float(d))))
                    if w in word_to_vec:
                        local_features.add("W_w2v: %s" % word_to_vec[w])
                if lexicon is not None:
                    if w in lexicon:
                        for feat in lexicon[w]:
                            local_features.add("W_lex: %s" % feat)
                    else:
                        local_features.add("W_lex: N/A")
                features.append(local_features)
        return features

    def _get_latent_features(self, words, start, beam, i):
        """"""
        features = set()
        global_i = start + i
        tags = ["<START-2>", "<START-1>"] + beam
        mapping = self.mapping
        j = i + 2
        if i >= 1:
            features.add("P1_word, P1_pos: %s, %s" % (words[global_i - 1], tags[j - 1]))
            if mapping is not None:
                features.add("P1_word, P1_wc: %s, %s" % (words[global_i - 1], mapping[tags[j - 1]]))
        if i >= 2:
            features.add("P2_word, P2_pos: %s, %s" % (words[global_i - 2], tags[j - 2]))
            if mapping is not None:
                features.add("P2_word, P2_wc: %s, %s" % (words[global_i - 2], mapping[tags[j - 2]]))
        features.add("P1_pos: %s" % tags[j - 1])
        features.add("P2_pos: %s" % tags[j - 2])
        features.add("P2_pos, P1_pos: %s, %s" % (tags[j - 2], tags[j - 1]))
        features.add("P1_pos, W_word: %s, %s" % (tags[j - 1], words[global_i]))
        if mapping is not None:
            features.add("P1_wc: %s" % mapping[tags[j - 1]])
            features.add("P2_wc: %s" % mapping[tags[j - 2]])
            features.add("P2_wc, P1_wc: %s, %s" % (mapping[tags[j - 2]], mapping[tags[j - 1]]))
            features.add("P1_wc, W_word: %s, %s" % (mapping[tags[j - 1]], words[global_i]))
        return features

    @staticmethod
    @functools.lru_cache(maxsize=10240)
    def _word_shape(word):
        if len(word) >= 100:
            return "LONG"
        shape = []
        last = ""
        shape_char = ""
        seq = 0
        for c in word:
            if c.isalpha():
                if c.isupper():
                    shape_char = "X"
                else:
                    shape_char = "x"
            elif c.isdigit():
                shape_char = "d"
            else:
                shape_char = c
            if shape_char == last:
                seq += 1
            else:
                seq = 0
                last = shape_char
            if seq < 4:
                shape.append(shape_char)
        return "".join(shape)

    @staticmethod
    @functools.lru_cache(maxsize=10240)
    def _word_flags(word, prefix):
        """"""
        email = re.compile(r"^[[:alnum:].%+-]+(?:@| \[?at\]? )[[:alnum:].-]+(?:\.| \[?dot\]? )[[:alpha:]]{2,}$", re.IGNORECASE)
        tag = re.compile(r"^</?[^>]+>$")
        url = re.compile(r"^(?:(?:(?:https?|ftp|svn)://|(?:https?://)?www\.).+)|(?:[\w./-]+\.(?:de|com|org|net|edu|info|jpg|png|gif|log|txt)(?:-\w+)?)$", re.IGNORECASE)
        mention = re.compile(r"^@\w+$")
        hashtag = re.compile(r"^#\w+$")
        action_word = re.compile(r"^[*+][^*]+[*]$")
        punctuation = re.compile(r'^[](){}.!?…<>%‰€$£₤¥°@~*„“”‚‘"\'`´»«›‹,;:/*+=&%§~#^−–-]+$')
        ordinal = re.compile(r"^(?:\d+\.)+$")
        number = re.compile(r"""(?<!\w)
                                (?:[−+-]?              # optional sign
                                  \d*                  # optional digits before decimal point
                                  [.,]?                # optional decimal point
                                  \d+                  # digits
                                  (?:[eE][−+-]?\d+)?   # optional exponent
                                  |
                                  \d+[\d.,]*\d+)
                                (?![.,]?\d)""", re.VERBOSE)
        emoticon_set = set(["(-.-)", "(T_T)", "(♥_♥)", ")':", ")-:",
                            "(-:", ")=", ")o:", ")x", ":'C", ":/", ":<",
                            ":C", ":[", "=(", "=)", "=D", "=P", ">:",
                            "D':", "D:", "\:", "]:", "x(", "^^", "o.O",
                            "oO", "\O/", "\m/", ":;))", "_))", "*_*",
                            "._.", ":wink:", ">_<", "*<:-)", ":!:",
                            ":;-))"])
        emoticon_list = sorted(emoticon_set, key=len, reverse=True)
        emoticon = re.compile(r"""^(?:(?:[:;]|(?<!\d)8)           # a variety of eyes, alt.: [:;8]
                                    [-'oO]?                       # optional nose or tear
                                    (?: \)+ | \(+ | [*] | ([DPp])\1*(?!\w)))   # a variety of mouths
                               """ +
                              r"|" +
                              r"(?:xD+|XD+)" +
                              r"|" +
                              r"([:;])[ ]+([()])" +
                              r"|" +
                              r"\^3"
                              r"|" +
                              r"emojiQ[[:alpha:]][3,}"
                              r"|" +
                              r"[\u2600-\u27BF\U0001F300-\U0001f64f\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]" +  # Unicode emoticons and other symbols
                              r"|" +
                              r"|".join([re.escape(_) for _ in emoticon_list]) +
                              r"$", re.VERBOSE)
        flags = set()
        if word.isalpha():
            flags.add("%s_isalpha" % prefix)
        if word.isnumeric():
            flags.add("%s_isnumeric" % prefix)
        if word.islower():
            flags.add("%s_islower" % prefix)
        if word.isupper():
            flags.add("%s_isupper" % prefix)
        if word.istitle():
            flags.add("%s_istitle" % prefix)
        if email.search(word):
            flags.add("%s_isemail" % prefix)
        if tag.search(word):
            flags.add("%s_istag" % prefix)
        if url.search(word):
            flags.add("%s_isurl" % prefix)
        if mention.search(word):
            flags.add("%s_ismention" % prefix)
        if hashtag.search(word):
            flags.add("%s_ishashtag" % prefix)
        if action_word.search(word):
            flags.add("%s_isactword" % prefix)
        if emoticon.search(word):
            flags.add("%s_isemoticon" % prefix)
        if punctuation.search(word):
            flags.add("%s_ispunct" % prefix)
        if ordinal.search(word):
            flags.add("%s_isordinal" % prefix)
        if number.search(word):
            flags.add("%s_isnumber" % prefix)
        return flags
