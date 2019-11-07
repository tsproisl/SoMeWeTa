#!/usr/bin/env python3

import argparse
import io
import itertools
import logging
import math
import multiprocessing
import os
import statistics
import threading
import time

from someweta import utils
from someweta import ASPTagger
from someweta.version import __version__


class Sentinel:
    pass


def arguments():
    """Process command line arguments."""
    parser = argparse.ArgumentParser(description="An averaged perceptron part-of-speech tagger")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", type=os.path.abspath, help="Train the tagger on the input corpus and write the model to the specified file")
    group.add_argument("--tag", type=os.path.abspath, help="Tag the input corpus using the specified model")
    group.add_argument("--evaluate", type=os.path.abspath, help="Evaluate the performance of the specified model on the input corpus")
    group.add_argument("--crossvalidate", action="store_true", help="Evaluate tagger performance via 10-fold cross-validation on the input corpus")
    parser.add_argument("--brown", type=argparse.FileType("r"), help="""Brown clusters (paths output file
                        produced by wcluster (https://github.com/percyliang/brown-cluster)); optional and only for
                        training or cross-validation""")
    parser.add_argument("--w2v", type=argparse.FileType("r"), help="Word2Vec vectors; optional and only for training or cross-validation")
    parser.add_argument("--lexicon", type=os.path.abspath, help="Additional full-form lexicon; optional and only for training or cross-validation")
    parser.add_argument("--mapping", type=os.path.abspath, help="Additional mapping to coarser tagset; optional and only for tagging, evaluating or cross-validation")
    parser.add_argument("--ignore-tag", help="Ignore this tag (useful for partial annotation); optional and only for training, evaluating or cross-validation")
    parser.add_argument("--prior", type=os.path.abspath, help="Prior weights, i.e. a model trained on another corpus; optional and only for training or cross-validation")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="Only for training or cross-validation: Number of iterations; default: 10")
    parser.add_argument("-b", "--beam-size", type=int, default=5, help="Size of the search beam; default: 5")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Run N worker processes (up to the number of CPUs) to speed up tagging.")
    parser.add_argument("-x", "--xml", action="store_true", help="The input is an XML file. We assume that each tag is on a separate line. Otherwise the format is the same as for regular files with respect to tag and sentence delimiters.")
    parser.add_argument("--progress", action="store_true", help="Show progress when tagging a file.")
    parser.add_argument("-v", "--version", action="version", version="SoMeWeTa %s" % __version__, help="Output version information and exit.")
    parser.add_argument("CORPUS", type=argparse.FileType("r", encoding="utf-8"),
                        help="""Input corpus (UTF-8-encoded). Path to a file or "-" for STDIN. Format for training,
                             evaluation and cross-validation: One
                             token-pos pair per line, separated by a
                             tab; sentences delimited by an empty
                             line. Format for tagging: One token per
                             line; sentences delimited by an empty
                             line.""")
    return parser.parse_args()


def evaluate_fold(args):
    i, beam_size, iterations, lexicon, mapping, brown_clusters, word_to_vec, words, tags, lengths, sentence_ranges, div, mod = args
    asptagger = ASPTagger(beam_size, iterations, lexicon, mapping, brown_clusters, word_to_vec)
    test_ranges = sentence_ranges[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)]
    test_start = test_ranges[0][0]
    test_end = test_ranges[-1][0] + test_ranges[-1][1]
    test_lengths = lengths[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)]
    train_lengths = lengths[:i * div + min(i, mod)] + lengths[(i + 1) * div + min(i + 1, mod):]
    test_words = words[test_start:test_end]
    test_tags = tags[test_start:test_end]
    train_words = words[:test_start] + words[test_end:]
    train_tags = tags[:test_start] + tags[test_end:]
    asptagger.train(train_words, train_tags, train_lengths)
    accuracy, accuracy_iv, accuracy_oov, coarse_accuracy, coarse_accuracy_iv, coarse_accuracy_oov = asptagger.evaluate(test_words, test_tags, test_lengths)
    logging.info("Accuracy: %.2f%%" % (accuracy * 100,))
    if coarse_accuracy is not None:
        logging.info("Accuracy on mapped tagset: %.2f%%" % (coarse_accuracy * 100,))
    return accuracy, accuracy_iv, accuracy_oov, coarse_accuracy, coarse_accuracy_iv, coarse_accuracy_oov


def fill_input_queue(input_queue, corpus, processes, sentinel, xml=False):
    """"""
    corpus_size = 0
    if xml:
        for i, (words, length, lines, word_indexes) in enumerate(utils.iter_xml(corpus, tagged=False)):
            corpus_size += length
            input_queue.put((i, words, lines, word_indexes))
    else:
        for i, (words, length) in enumerate(utils.iter_corpus(corpus, tagged=False)):
            corpus_size += length
            input_queue.put((i, words))
    for proc in range(processes):
        input_queue.put(sentinel)
    input_queue.put(corpus_size)


def process_input_queue(func, input_queue, output_queue, sentinel, xml=False):
    """"""
    while True:
        data = input_queue.get()
        if isinstance(data, Sentinel):
            break
        if xml:
            i, words, lines, word_indexes = data
            result = func(words)
            output_queue.put((i, result, lines, word_indexes))
        else:
            i, words = data
            result = func(words)
            output_queue.put((i, result))
    output_queue.put(sentinel)


def parallel_tagging(args, asptagger, progress=None, xml=False):
    """"""
    def output_result(data, xml, progress):
        if xml:
            i, result, lines, word_indexes = data
            print("\n".join(utils.add_pos_to_xml(result, lines, word_indexes)), "\n", sep="")
        else:
            i, result = data
            print("\n".join(["\t".join(t) for t in result]), "\n", sep="")
        if progress is not None:
            progress.update(len(result))

    sentinel = Sentinel()
    processes = min(args.parallel, multiprocessing.cpu_count())
    input_queue = multiprocessing.Queue(maxsize=processes * 100)
    output_queue = multiprocessing.Queue(maxsize=processes * 100)
    producer = threading.Thread(target=fill_input_queue, args=(input_queue, args.CORPUS, processes, sentinel, xml))
    with multiprocessing.Pool(processes=processes, initializer=process_input_queue, initargs=(asptagger.tag_sentence, input_queue, output_queue, sentinel, xml)):
        producer.start()
        observed_sentinels = 0
        current = 0
        cached_results = {}
        while True:
            data = output_queue.get()
            if isinstance(data, Sentinel):
                observed_sentinels += 1
                if observed_sentinels == processes:
                    break
                else:
                    continue
            i = data[0]
            if i == current:
                output_result(data, xml, progress)
                current += 1
            else:
                cached_results[i] = data
            while current in cached_results:
                output_result(cached_results[current], xml, progress)
                del cached_results[current]
                current += 1
        corpus_size = input_queue.get()
        producer.join()
        return corpus_size


def get_number_of_tokens(queue, corpus, xml):
    """"""
    n = 0
    try:
        corpus.seek(0)
    except io.UnsupportedOperation:
        logging.info("Input stream is not seekable, cannot determine ETA.")
        logging.info("If you want SoMeWeTa to estimate the remaining time, please provide the input as a regular file.")
        queue.put(None)
        return
    except ValueError:
        logging.info("Input stream is not seekable, cannot determine ETA.")
        logging.info("If you want SoMeWeTa to estimate the remaining time, please provide the input as a regular file.")
        queue.put(None)
        return
    if xml:
        for words, length, lines, word_indexes in utils.iter_xml(corpus, tagged=False):
            n += length
    else:
        for words, length in utils.iter_corpus(corpus, tagged=False):
            n += length
    corpus.seek(0)
    queue.put(n)


def main():
    args = arguments()
    lexicon, mapping, brown_clusters, word_to_vec = None, None, None, None
    if args.progress:
        if args.tag:
            n_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=get_number_of_tokens, args=(n_queue, args.CORPUS, args.xml))
            p.start()
        else:
            logging.warning("Currently, the --progress option is only available for tagging, i.e. in combination with --tag.")
    if args.mapping and (args.tag or args.evaluate or args.crossvalidate):
        mapping = utils.read_mapping(args.mapping)
    if args.lexicon and (args.train or args.crossvalidate):
        lexicon = utils.read_lexicon(args.lexicon)
    if args.brown and (args.train or args.crossvalidate):
        brown_clusters = utils.read_brown_clusters(args.brown)
    if args.w2v and (args.train or args.crossvalidate):
        word_to_vec = utils.read_word2vec_vectors(args.w2v)
    asptagger = ASPTagger(args.beam_size, args.iterations, lexicon, mapping, brown_clusters, word_to_vec, args.ignore_tag)
    if args.prior and (args.train or args.crossvalidate):
        asptagger.load_prior_model(args.prior)
    if args.train:
        if args.xml:
            words, tags, lengths = utils.read_tagged_xml(args.CORPUS)
        else:
            words, tags, lengths = utils.read_corpus(args.CORPUS, tagged=True)
        asptagger.train(words, tags, lengths)
        asptagger.save(args.train)
    elif args.tag:
        prog = None
        asptagger.load(args.tag)
        if args.progress:
            n = n_queue.get()
            p.join()
            prog = utils.Progress(length=n, rate=1000)
        t0 = time.perf_counter()
        if args.xml:
            if args.parallel > 1:
                corpus_size = parallel_tagging(args, asptagger, prog, xml=True)
            else:
                corpus_size = 0
                for words, length, lines, word_indexes in utils.iter_xml(args.CORPUS, tagged=False):
                    corpus_size += length
                    sentence = asptagger.tag_sentence(words)
                    print("\n".join(utils.add_pos_to_xml(sentence, lines, word_indexes)), "\n", sep="")
                    if args.progress:
                        prog.update(length)
        else:
            if args.parallel > 1:
                corpus_size = parallel_tagging(args, asptagger, prog)
            else:
                corpus_size = 0
                for words, length in utils.iter_corpus(args.CORPUS, tagged=False):
                    corpus_size += length
                    sentence = asptagger.tag_sentence(words)
                    print("\n".join(["\t".join(t) for t in sentence]), "\n", sep="")
                    if args.progress:
                        prog.update(length)
        if args.progress:
            prog.finalize()
        t1 = time.perf_counter()
        logging.info("Tagged %d tokens in %s (%d tokens/s)" % (corpus_size, utils.int2str(t1 - t0), corpus_size / (t1 - t0)))
    elif args.evaluate:
        asptagger.load(args.evaluate)
        if args.xml:
            words, tags, lengths = utils.read_tagged_xml(args.CORPUS)
        else:
            words, tags, lengths = utils.read_corpus(args.CORPUS, tagged=True)
        accuracy, accuracy_iv, accuracy_oov, coarse_accuracy, coarse_accuracy_iv, coarse_accuracy_oov = asptagger.evaluate(words, tags, lengths)
        print("Accuracy: %.2f%%; IV: %.2f%%; OOV: %.2f%%" % (accuracy * 100, accuracy_iv * 100, accuracy_oov * 100))
        if coarse_accuracy is not None:
            print("Accuracy on mapped tagset: %.2f%%; IV: %.2f%%; OOV: %.2f%%" % (coarse_accuracy * 100, coarse_accuracy_iv * 100, coarse_accuracy_oov * 100))
    elif args.crossvalidate:
        if args.xml:
            words, tags, lengths = utils.read_tagged_xml(args.CORPUS)
        else:
            words, tags, lengths = utils.read_corpus(args.CORPUS, tagged=True)
        sentence_ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        div, mod = divmod(len(sentence_ranges), 10)
        with multiprocessing.Pool() as pool:
            accs = pool.map(evaluate_fold, zip(range(10), itertools.repeat(args.beam_size), itertools.repeat(args.iterations), itertools.repeat(lexicon), itertools.repeat(mapping), itertools.repeat(brown_clusters), itertools.repeat(word_to_vec), itertools.repeat(words), itertools.repeat(tags), itertools.repeat(lengths), itertools.repeat(sentence_ranges), itertools.repeat(div), itertools.repeat(mod)))
        accuracies, accuracies_iv, accuracies_oov, coarse_accuracies, coarse_accuracies_iv, coarse_accuracies_oov = zip(*accs)
        mean_accuracy = statistics.mean(accuracies)
        # 2.26 is the approximate value of the 97.5 percentile point
        # of the t distribution with 9 degrees of freedom. We use the
        # t distribution instead of the standard normal distribution
        # (= 1.96) because the population standard deviation is
        # unknown and because we have a small sample size (10).
        confidence = 2.26 * statistics.stdev(accuracies) / math.sqrt(10)
        print("Mean accuracy and 95%% confidence interval: %.2f%% ±%.2f" % (mean_accuracy * 100, confidence * 100))
        if coarse_accuracies[0] is not None:
            coarse_mean_accuracy = statistics.mean(coarse_accuracies)
            coarse_confidence = 2.26 * statistics.stdev(coarse_accuracies) / math.sqrt(10)
            print("Mean accuracy and 95%% confidence interval on mapped tagset: %.2f%% ±%.2f" % (coarse_mean_accuracy * 100, coarse_confidence * 100))
