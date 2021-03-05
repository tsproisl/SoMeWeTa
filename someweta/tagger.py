#!/usr/bin/env python3

import base64
import functools
import gzip
import html
import json
import math

import numpy as np
import regex as re

from someweta.averaged_structured_perceptron import AveragedStructuredPerceptron


class ASPTagger(AveragedStructuredPerceptron):
    """A part-of-speech tagger based on the averaged structured
    perceptron.

    """
    def __init__(self, beam_size=5, iterations=10, lexicon=None, mapping=None, brown_clusters=None, word_to_vec=None, ignore_tag=None):
        super().__init__(beam_size=beam_size, beam_history=2, iterations=iterations, latent_features=None, ignore_target=ignore_tag)
        self.vocabulary = set()
        self.lexicon = lexicon
        self.mapping = mapping
        if self.mapping is not None and self.ignore_target is not None:
            self.mapping[self.ignore_target] = self.ignore_target
        self.brown_clusters = brown_clusters
        self.word_to_vec = word_to_vec
        self.email = re.compile(r"^[\w.%+-]+(?:@| \[?at\]? )[\w.-]+(?:\.| \[?dot\]? )[a-z]{2,}$", re.IGNORECASE)
        self.xmltag = re.compile(r"^</?[^>]+>$")
        self.url = re.compile(r"^" +
                              r"(?:(?:(?:https?|ftp|svn)://|(?:https?://)?www\.).+)" +  # anything that starts with http, https, ftp, svn or www
                              r"|" +
                              r"(?:[\w./-]+\.(?:de|at|ch|com|org|net|edu|info|gov|jpg|png|gif|log|txt|xlsx?|docx?|pptx?|pdf)(?:-\w+)?)" +  # anything with those TLDs or file extensions
                              r"|" +
                              r"(?:/?[rlu](?:/\w+)+/?)" +  # a Reddit link
                              r"$", re.IGNORECASE)
        self.mention = re.compile(r"^@\w+$")
        self.hashtag = re.compile(r"^#\w+$")
        self.action_word = re.compile(r"^[*+][^*]+[*]$")
        self.punctuation = re.compile(r'^[][(){},;:.!?…„“”‚‘’"\'`´»«›‹/−–-]+$')
        self.ordinal = re.compile(r"^(?:\d+\.)+$")
        self.number = re.compile(r"""^
                                (?:[−+-]?              # optional sign
                                  \d*                  # optional digits before decimal point
                                  [.,]?                # optional decimal point
                                  \d+                  # digits
                                  (?:[eE][−+-]?\d+)?   # optional exponent
                                  |
                                  \d+[\d.,]*\d+)
                                $""", re.VERBOSE)
        emoticon_set = set(["(-.-)", "(T_T)", "(♥_♥)", ")':", ")-:",
                            "(-:", ")=", ")o:", ")x", ":'C", ":/",
                            ":<", ":C", ":[", "=(", "=)", "=D", "=P",
                            ">:", "\\:", "]:", "x(", "^^", "o.O",
                            "\\O/", "\\m/", ":;))", "_))", "*_*",
                            "._.", ">_<", "*<:-)", ":!:", ":;-))",
                            "x'D", ":^)", "<3"])
        reddit_emoticons = set(["Ä", "Ö", "Ü"])
        emoticon_set.update(reddit_emoticons)
        # From https://textfac.es/
        textfaces_space = set(['⚆ _ ⚆', '˙ ͜ʟ˙', '◔ ⌣ ◔', '( ﾟヮﾟ)', '(• ε •)',
                               '(づ￣ ³￣)づ', '♪~ ᕕ(ᐛ)ᕗ', '\\ (•◡•) /', '( ಠ ͜ʖರೃ)',
                               '( ⚆ _ ⚆ )', '(▀̿Ĺ̯▀̿ ̿)', '༼ つ ◕_◕ ༽つ', '༼ つ ಥ_ಥ ༽つ',
                               '( ͡° ͜ʖ ͡°)', '( ͡°╭͜ʖ╮͡° )', '(╯°□°）╯︵ ┻━┻',
                               '( ͡ᵔ ͜ʖ ͡ᵔ )', '┬──┬ ノ( ゜-゜ノ)', '┬─┬ノ( º _ ºノ)',
                               '(ง ͠° ͟ل͜ ͡°)ง', '(͡ ͡° ͜ つ ͡͡°)', "﴾͡๏̯͡๏﴿ O'RLY?",
                               '（╯°□°）╯︵( .o.)', '(° ͡ ͜ ͡ʖ ͡ °)', '┬─┬ ︵ /(.□. ）',
                               '(/) (°,,°) (/)', '(☞ﾟヮﾟ)☞ ☜(ﾟヮﾟ☜)', '| (• ◡•)| (❍ᴥ❍ʋ)',
                               '༼ つ ͡° ͜ʖ ͡° ༽つ', '(╯°□°)╯︵ ʞooqǝɔɐɟ', '┻━┻ ︵ヽ(`Д´)ﾉ︵ ┻━┻',
                               '┬┴┬┴┤ ͜ʖ ͡°) ├┬┴┬┴', '(ó ì_í)=óò=(ì_í ò)',
                               '(•_•) ( •_•)>⌐■-■ (⌐■_■)', '(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧ ✧ﾟ･: *ヽ(◕ヮ◕ヽ)',
                               '[̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅]', '/╲/\\╭( ͡° ͡° ͜ʖ ͡° ͡°)╮/\\╱\\',
                               '( ͡°( ͡° ͜ʖ( ͡° ͜ʖ ͡°)ʖ ͡°) ͡°)', '(._.) ( l: ) ( .-. ) ( :l ) (._.)',
                               "̿ ̿ ̿'̿'\\̵͇̿̿\\з=(•_•)=ε/̵͇̿̿/'̿'̿ ̿", '༼ ºل͟º ༼ ºل͟º ༼ ºل͟º ༽ ºل͟º ༽ ºل͟º ༽',
                               "̿'̿'\\̵͇̿̿\\з=( ͠° ͟ʖ ͡°)=ε/̵͇̿̿/'̿̿ ̿ ̿ ̿ ̿ ̿",
                               "̿̿ ̿̿ ̿̿ ̿'̿'\\̵͇̿̿\\з= ( ▀ ͜͞ʖ▀) =ε/̵͇̿̿/’̿’̿ ̿ ̿̿ ̿̿ ̿̿"])
        textfaces_emoji = set(['♥‿♥', '☼.☼', '≧☉_☉≦', '(°ロ°)☝', '(☞ﾟ∀ﾟ)☞', '☜(˚▽˚)☞', '☜(⌒▽⌒)☞', '(☞ຈل͜ຈ)☞', 'ヾ(⌐■_■)ノ♪'])
        textfaces_wo_emoji = set(['=U', 'ಠ_ಠ', '◉_◉', 'ಥ_ಥ', ":')", 'ಠ⌣ಠ', 'ಠ~ಠ', 'ಠ_ಥ', 'ಠ‿↼', 'ʘ‿ʘ', 'ಠoಠ', 'ರ_ರ', '◔̯◔', '¬_¬', 'ب_ب', '°Д°', '^̮^', '^̮^', '^̮^', '>_>', '^̮^', '^̮^', 'ಠ╭╮ಠ', '(>ლ)', 'ʕ•ᴥ•ʔ', '(ಥ﹏ಥ)', '(ᵔᴥᵔ)', '(¬‿¬)', '⌐╦╦═─', '(•ω•)', '(¬_¬)', '｡◕‿◕｡', '(ʘ‿ʘ)', '٩◔̯◔۶', '(>人<)', '(~_^)', '(^̮^)', '(･.◤)', '(◕‿◕✿)', '｡◕‿‿◕｡', '(─‿‿─)', '(；一_一)', "(ʘᗩʘ')", '(✿´‿`)', 'ლ(ಠ益ಠლ)', '~(˘▾˘~)', '(~˘▾˘)~', '(｡◕‿◕｡)', '(っ˘ڡ˘ς)', 'ლ(´ڡ`ლ)', 'ƪ(˘⌣˘)ʃ', '(´・ω・`)', '(ღ˘⌣˘ღ)', '(▰˘◡˘▰)', '〆(・∀・＠)', '༼ʘ̚ل͜ʘ̚༽', 'ᕙ(⇀‸↼‶)ᕗ', 'ᕦ(ò_óˇ)ᕤ', '(｡◕‿‿◕｡)', 'ヽ༼ຈل͜ຈ༽ﾉ', '(ง°ل͜°)ง', '╚(ಠ_ಠ)=┐', '(´・ω・)っ由', 'Ƹ̵̡Ӝ̵̨̄Ʒ', '¯\\_(ツ)_/¯', '▄︻̷̿┻̿═━一', "(ง'̀-'́)ง", '¯\\(°_o)/¯', '｡゜(｀Д´)゜｡', '(づ｡◕‿‿◕｡)づ', '(;´༎ຶД༎ຶ`)', '(ノಠ益ಠ)ノ彡┻━┻', 'ლ,ᔑ•ﺪ͟͠•ᔐ.ლ', '(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧', '┬┴┬┴┤(･_├┬┴┬┴', '[̲̅$̲̅(̲̅5̲̅)̲̅$̲̅]'])
        emoticon_list = sorted(emoticon_set | textfaces_space | textfaces_emoji | textfaces_wo_emoji, key=len, reverse=True)
        self.emoticon = re.compile(r"""^(?:(?:(?:[:;]|(?<!\d)8)           # a variety of eyes, alt.: [:;8]
                                    [-'oO]?                       # optional nose or tear
                                    (?: \)+ | \(+ | [*] | ([DPp])\1*(?!\w)))   # a variety of mouths
                                    """ +
                                   r"|" +
                                   r"(?:[Xx]D+)" +
                                   r"|" +
                                   r"([:;])[ ]+([()])" +
                                   r"|" +
                                   r"\^3" +
                                   r"|" +
                                   r"(?:(?:D'?:|oO))" +
                                   r"|" +
                                   r"(?::\w+:)" +   # Textual representations of emojis: :smile:, etc. We don't want to match times: 08:30:00
                                   r"|" +
                                   r"|".join([re.escape(_) for _ in emoticon_list]) +
                                   r")$", re.VERBOSE)
        # Unicode emoticons and other symbols
        self.unicode_flags = re.compile(r"^\p{Regional_Indicator}{2}$")
        # self.emoji = re.compile(r"^[\u2600-\u27BF\uFE0E\uFE0F\U0001F300-\U0001f64f\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]$")
        self.emoji = re.compile(r"[\p{Extended_Pictographic}\p{Emoji_Presentation}\uFE0F\u2600-\u27BF]")

    def train(self, words, tags, lengths):
        """"""
        lower_words = [w.lower() for w in words]
        self.latent_features = functools.partial(self._get_latent_features, lower_words)
        self.vocabulary.update(set(words))
        # self.vocabulary.update(set(lower_words))
        # <OOV>
        # # vocabulary = all lower case word forms except hapax legomena
        # self.vocabulary.update(set(k for k, v in collections.Counter(lower_words).items() if v > 1))
        # </OOV>
        X = self._get_static_features(words, lengths)
        self.fit(X, tags, lengths)

    def tag(self, words, lengths):
        """"""
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in words])
        X = self._get_static_features(words, lengths)
        tags = self.predict(X, lengths)
        start = 0
        for length, local_tags in zip(lengths, tags):
            local_words = words[start:start + length]
            start += length
            if self.mapping is not None:
                yield zip(local_words, local_tags, (self.mapping[lt] for lt in local_tags))
            else:
                yield zip(local_words, local_tags)

    def tag_sentence(self, sentence):
        """"""
        sentence_length = [len(sentence)]
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in sentence])
        X = self._get_static_features(sentence, sentence_length)
        tags = list(self.predict(X, sentence_length))[0]
        if self.mapping is not None:
            return list(zip(sentence, tags, (self.mapping[lt] for lt in tags)))
        else:
            return list(zip(sentence, tags))

    def tag_xml_sentence(self, sentence):
        """Tag a sentence that contains XML tags in addition to the word
        tokens. For example the output of SoMaJo's tokenize_xml and
        split_xml methods.

        """
        word_indexes = [i for i, line in enumerate(sentence) if not (line.startswith("<") and line.endswith(">"))]
        words = [sentence[i] for i in word_indexes]
        words = [html.unescape(w) for w in words]
        tagged = self.tag_sentence(words)
        tags = {i: t[1:] for i, t in zip(word_indexes, tagged)}
        tagged_xml = []
        for idx, x in enumerate(sentence):
            tagged_xml.append((x,) + tags.get(idx, ()))
        return tagged_xml

    def evaluate(self, words, tags, lengths):
        """"""
        self.latent_features = functools.partial(self._get_latent_features, [w.lower() for w in words])
        X = self._get_static_features(words, lengths)
        # accuracy = self.score(X, tags, lengths)
        # return accuracy
        predicted = self.predict(X, lengths)
        correct, correct_iv, correct_oov = 0, 0, 0
        coarse_correct, coarse_correct_iv, coarse_correct_oov = 0, 0, 0
        total, total_iv, total_oov = 0, 0, 0
        start = 0
        for length, local_pred in zip(lengths, predicted):
            local_words = words[start:start + length]
            local_gold = tags[start:start + length]
            start += length
            for w, g, p in zip(local_words, local_gold, local_pred):
                if self.ignore_target is not None and g == self.ignore_target:
                    continue
                total += 1
                # <OOV>
                # if w.lower() in self.vocabulary:
                # </OOV>
                if w in self.vocabulary:
                    total_iv += 1
                    if g == p:
                        correct += 1
                        correct_iv += 1
                    if self.mapping is not None:
                        if self.mapping[g] == self.mapping[p]:
                            coarse_correct += 1
                            coarse_correct_iv += 1
                else:
                    total_oov += 1
                    if g == p:
                        correct += 1
                        correct_oov += 1
                    if self.mapping is not None:
                        if self.mapping[g] == self.mapping[p]:
                            coarse_correct += 1
                            coarse_correct_oov += 1

        accuracy = correct / total
        try:
            accuracy_iv = correct_iv / total_iv
        except ZeroDivisionError:
            accuracy_iv = 0
        try:
            accuracy_oov = correct_oov / total_oov
        except ZeroDivisionError:
            accuracy_oov = 0
        coarse_accuracy, coarse_accuracy_iv, coarse_accuracy_oov = None, None, None
        if self.mapping is not None:
            coarse_accuracy = coarse_correct / total
            try:
                coarse_accuracy_iv = coarse_correct_iv / total_iv
            except ZeroDivisionError:
                coarse_accuracy_iv = 0
            try:
                coarse_accuracy_oov = coarse_correct_oov / total_oov
            except ZeroDivisionError:
                coarse_accuracy_oov = 0
        return accuracy, accuracy_iv, accuracy_oov, coarse_accuracy, coarse_accuracy_iv, coarse_accuracy_oov

    def save(self, filename):
        """"""
        with gzip.open(filename, 'wb') as f:
            features = sorted(self.weights.keys())
            f.write("[\n".encode())
            f.write(json.dumps(list(self.vocabulary), ensure_ascii=False, indent=4).encode())
            f.write(",\n".encode())
            f.write(json.dumps(self.lexicon, ensure_ascii=False, indent=4).encode())
            f.write(",\n".encode())
            f.write(json.dumps(self.brown_clusters, ensure_ascii=False, indent=4).encode())
            f.write(",\n".encode())
            f.write(json.dumps(self.word_to_vec, ensure_ascii=False, indent=4).encode())
            f.write(",\n".encode())
            f.write(json.dumps(self.target_mapping, ensure_ascii=False, indent=4).encode())
            f.write(",\n".encode())
            f.write(str(self.target_size).encode())
            f.write(",\n".encode())
            f.write(json.dumps(features, ensure_ascii=False, indent=4).encode())
            f.write(",\n".encode())
            f.write("[\n".encode())
            for feat in features[:-1]:
                f.write('"'.encode())
                f.write(base64.b85encode(self.weights[feat].tobytes()))
                f.write('",\n'.encode())
            f.write('"'.encode())
            f.write(base64.b85encode(self.weights[features[-1]].tobytes()))
            f.write('"\n]\n'.encode())
            f.write("]\n".encode())

    def load(self, filename):
        """"""
        with gzip.open(filename, 'rb') as f:
            model = json.loads(f.read().decode())
            vocabulary, self.lexicon, self.brown_clusters, self.word_to_vec, self.target_mapping, self.target_size, features, weights = model
            self.vocabulary = set(vocabulary)
            self.weights = {f: np.frombuffer(base64.b85decode(w), np.float64).copy() for f, w in zip(features, weights)}

    def load_prior_model(self, prior):
        """"""
        with gzip.open(prior, 'rb') as f:
            model = json.loads(f.read().decode())
            self.vocabulary = set(model[0])
            self.target_mapping = model[4]
            self.target_size = model[5]
            features = model[6]
            weights = model[7]
            self.prior_weights = {f: np.frombuffer(base64.b85decode(w), np.float64).copy() for f, w in zip(features, weights)}

    def _get_static_features(self, words, lengths):
        """"""
        lexicon = self.lexicon
        brown_clusters = self.brown_clusters
        word_to_vec = self.word_to_vec
        # <OOV>
        # vocabulary = self.vocabulary
        # </OOV>
        features = []
        start = 0
        for length in lengths:
            sentence = words[start:start + length]
            start += length
            tokens = ["<START-2>", "<START-1>"] + [w.lower() for w in sentence] + ["<END+1>", "<END+2>"]
            for i, word in enumerate(sentence):
                j = i + 2
                local_features = []
                w = tokens[j]
                p1 = tokens[j - 1]
                p2 = tokens[j - 2]
                n1 = tokens[j + 1]
                n2 = tokens[j + 2]
                # constant bias feature acts like a prior
                local_features.append("bias")
                # rounded logarithm of word length
                local_features.append("W_loglength: %d" % round(math.log(len(word))))
                # current word
                local_features.append("W_word: %s" % w)
                # <OOV>
                # if w in vocabulary:
                #     local_features.append("W_word: %s" % w)
                # else:
                #     local_features.append("W_word: OOV")
                # </OOV>
                # next words
                local_features.append("N1_word: %s" % n1)
                # <OOV>
                # if n1 in vocabulary:
                #     local_features.append("N1_word: %s" % n1)
                # else:
                #     local_features.append("N1_word: OOV")
                # </OOV>
                local_features.append("N2_word: %s" % n2)
                # <OOV>
                # if n2 in vocabulary:
                #     local_features.append("N2_word: %s" % n2)
                # else:
                #     local_features.append("N2_word: OOV")
                # </OOV>
                # affixes
                local_features.append("W_prefix: %s" % w[:3])
                local_features.append("W_suffix: %s" % w[-3:])
                if i >= 1:
                    local_features.append("P1_suffix: %s" % p1[-3:])
                if length - i > 1:
                    local_features.append("N1_suffix: %s" % n1[-3:])
                # word shape
                local_features.append("W_shape: %s" % self._word_shape(word))
                # Flags
                if i >= 2:
                    local_features.extend(self._word_flags(p2, "P2"))
                if i >= 1:
                    local_features.extend(self._word_flags(p1, "P1"))
                local_features.extend(self._word_flags(w, "W"))
                if length - i > 1:
                    local_features.extend(self._word_flags(n1, "N1"))
                if length - i > 2:
                    local_features.extend(self._word_flags(n2, "N2"))
                # Brown clusters
                if brown_clusters is not None:
                    # P2, P1, W, N1, N2
                    if i >= 2:
                        bc, freq = brown_clusters.get(p2, ("N/A", 0))
                        local_features.append("P2_brown: %s" % bc)
                    if i >= 1:
                        bc, freq = brown_clusters.get(p1, ("N/A", 0))
                        local_features.append("P1_brown: %s" % bc)
                    bc, freq = brown_clusters.get(w, ("N/A", 0))
                    local_features.append("W_brown: %s" % bc)
                    local_features.append("W_logfreq: %d" % freq)
                    if length - i > 1:
                        bc, freq = brown_clusters.get(n1, ("N/A", 0))
                        local_features.append("N1_brown: %s" % bc)
                    if length - i > 2:
                        bc, freq = brown_clusters.get(n2, ("N/A", 0))
                        local_features.append("N2_brown: %s" % bc)
                if word_to_vec is not None:
                    # if w in word_to_vec:
                    #     for i, d in enumerate(word_to_vec[w]):
                    #         local_features.append("W_w2v_%d: %d" % (i, round(float(d))))
                    if w in word_to_vec:
                        local_features.append("W_w2v: %s" % word_to_vec[w])
                if lexicon is not None:
                    if w in lexicon:
                        for feat in lexicon[w]:
                            local_features.append("W_lex: %s" % feat)
                    else:
                        local_features.append("W_lex: N/A")
                features.append(local_features)
        return features

    def _get_latent_features(self, words, start, beam, i):
        """"""
        # <OOV>
        # vocabulary = self.vocabulary
        # </OOV>
        features = []
        global_i = start + i
        tags = ["<START-2>", "<START-1>"] + beam
        j = i + 2
        if i >= 1:
            features.append("P1_word, P1_pos: %s, %s" % (words[global_i - 1], tags[j - 1]))
            # <OOV>
            # if words[global_i - 1] in vocabulary:
            #     features.append("P1_word, P1_pos: %s, %s" % (words[global_i - 1], tags[j - 1]))
            # else:
            #     features.append("P1_word, P1_pos: OOV, %s" % tags[j - 1])
            # </OOV>
        if i >= 2:
            features.append("P2_word, P2_pos: %s, %s" % (words[global_i - 2], tags[j - 2]))
            # <OOV>
            # if words[global_i - 2] in vocabulary:
            #     features.append("P2_word, P2_pos: %s, %s" % (words[global_i - 2], tags[j - 2]))
            # else:
            #     features.append("P2_word, P2_pos: OOV, %s" % tags[j - 2])
            # </OOV>
        features.append("P1_pos: %s" % tags[j - 1])
        features.append("P2_pos: %s" % tags[j - 2])
        features.append("P2_pos, P1_pos: %s, %s" % (tags[j - 2], tags[j - 1]))
        features.append("P1_pos, W_word: %s, %s" % (tags[j - 1], words[global_i]))
        # <OOV>
        # if words[global_i] in vocabulary:
        #     features.append("P1_pos, W_word: %s, %s" % (tags[j - 1], words[global_i]))
        # else:
        #     features.append("P1_pos, W_word: %s, OOV" % tags[j - 1])
        # </OOV>
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

    @functools.lru_cache(maxsize=10240)
    def _word_flags(self, word, prefix):
        """"""
        flags = []
        if word.isalpha():
            flags.append("%s_isalpha" % prefix)
        if word.isnumeric():
            flags.append("%s_isnumeric" % prefix)
        if word.islower():
            flags.append("%s_islower" % prefix)
        if word.isupper():
            flags.append("%s_isupper" % prefix)
        if word.istitle():
            flags.append("%s_istitle" % prefix)
        if self.email.search(word):
            flags.append("%s_isemail" % prefix)
        if self.xmltag.search(word):
            flags.append("%s_istag" % prefix)
        if self.url.search(word):
            flags.append("%s_isurl" % prefix)
        if self.mention.search(word):
            flags.append("%s_ismention" % prefix)
        if self.hashtag.search(word):
            flags.append("%s_ishashtag" % prefix)
        if self.action_word.search(word):
            flags.append("%s_isactword" % prefix)
        if self.emoticon.search(word):
            flags.append("%s_isemoticon" % prefix)
        if self.emoji.search(word):
            flags.append("%s_isemoji" % prefix)
        if self.punctuation.search(word):
            flags.append("%s_ispunct" % prefix)
        if self.ordinal.search(word):
            flags.append("%s_isordinal" % prefix)
        if self.number.search(word):
            flags.append("%s_isnumber" % prefix)
        return flags
