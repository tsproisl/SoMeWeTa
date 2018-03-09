#!/usr/bin/env python3

import collections
import itertools
import logging
import operator
import random

import numpy as np

from someweta import utils

Beam = collections.namedtuple("Beam", ["tags", "weight_sum", "features", "previous"])


class AveragedStructuredPerceptron:
    """An averaged structured perceptron.

    The perceptron algorithm is due to Rosenblatt (1958). Freund and
    Schapire (1999: 292) found "that voting and averaging perform
    better than using the last vector". We implement averaging.
    Collins (2002) introduced the structured perceptron and Collins
    and Roark (2004) suggested the early update strategy.

    """
    def __init__(self, beam_size, beam_history, iterations, latent_features, prior_weights=None, ignore_target=None):
        self.beam_size = beam_size
        self.beam_history = beam_history
        self.iterations = iterations
        self.latent_features = latent_features
        self.prior_weights = prior_weights
        self.ignore_target = ignore_target
        # self.weights = collections.defaultdict(lambda: collections.defaultdict(float))
        # self.weights_c = collections.defaultdict(lambda: collections.defaultdict(float))
        self.target_mapping = {}
        self.reverse_mapping = None
        self.target_size = 0
        self.ignore_target_mapping = None
        self.weights = {}
        self.weights_c = {}

    def fit(self, X, y, lengths):
        """"""
        targets = collections.Counter(y)
        former_target_size = self.target_size
        for target, freq in reversed(targets.most_common()):
            if target not in self.target_mapping and target != self.ignore_target:
                self.target_mapping[target] = self.target_size
                self.target_size += 1
        if self.ignore_target is not None:
            self.ignore_target_mapping = self.target_size
        y = [self.target_mapping.get(target, self.ignore_target_mapping) for target in y]
        if self.target_size > former_target_size and self.prior_weights is not None:
            for feat in self.prior_weights:
                self.prior_weights[feat].resize((self.target_size,))
        counter = 0
        ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        for it in range(self.iterations):
            total, incorrect, early_update = 0, 0, 0
            for start, length in ranges:
                local_X = X[start:start + length]
                local_y = y[start:start + length]
                predicted, features = self._beam_search(local_X, start, local_y)
                assert type(predicted) == type(y)
                if len(predicted) != len(local_y):
                    early_update += 1
                if self.ignore_target is not None:
                    erroneous = any(p != g and g != self.ignore_target_mapping for p, g in zip(predicted, local_y))
                else:
                    erroneous = predicted != local_y
                if erroneous:
                    incorrect += 1
                    self._update(local_y, predicted, features, counter)
                counter += len(predicted)
                total += 1
            # random.seed(it)
            random.shuffle(ranges)
            correct = total - incorrect
            logging.info("Iteration %d: %d/%d = %.2f%% (%d early update)" % (it, correct, total, (correct / total) * 100, early_update))
        for feat in self.weights:
            self.weights[feat] -= self.weights_c[feat] / counter
        if self.prior_weights is not None:
            for feat in self.prior_weights:
                if feat not in self.weights:
                    self.weights[feat] = self.prior_weights[feat]
                else:
                    self.weights[feat] += self.prior_weights[feat]

    def predict(self, X, lengths):
        """"""
        if self.reverse_mapping is None:
            self.reverse_mapping = {v: k for k, v in self.target_mapping.items()}
        ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        for start, length in ranges:
            local_X = X[start:start + length]
            predicted, features = self._beam_search(local_X, start)
            predicted = [self.reverse_mapping[p] for p in predicted]
            yield predicted

    def score(self, X, y, lengths):
        """"""
        if self.reverse_mapping is None:
            self.reverse_mapping = {v: k for k, v in self.target_mapping.items()}
        predicted = []
        ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        for start, length in ranges:
            local_X = X[start:start + length]
            local_pred, features = self._beam_search(local_X, start)
            local_pred = [self.reverse_mapping[p] for p in local_pred]
            predicted.extend(local_pred)
        accuracy = utils.evaluate(y, predicted, self.ignore_target)
        coarse_accuracy = None
        if self.mapping is not None:
            coarse_y = [self.mapping[t] for t in y]
            coarse_predicted = [self.mapping[t] for t in predicted]
            coarse_accuracy = utils.evaluate(coarse_y, coarse_predicted, self.ignore_target)
        return accuracy, coarse_accuracy

    @staticmethod
    def _extract_feature_sequence(beam):
        """"""
        sequence = [beam.features]
        while beam.previous is not None:
            sequence.append(beam.features)
            beam = beam.previous
        return sequence[::-1]

    def _beam_search(self, X, start, y=None):
        """"""
        beams = [Beam([], 0, [], None)]
        gold_tags = []
        for i, static_features in enumerate(X):
            agenda = {}
            weight_sum = self._predict_static(static_features)
            for beam in beams:
                latent_features = self.latent_features(start, beam.tags, i)
                features = static_features + latent_features
                for prediction, weight in self._predict_latent(latent_features, weight_sum):
                    tags = beam.tags + [prediction]
                    history = tuple(tags[-self.beam_history:])
                    new_weight_sum = beam.weight_sum + weight
                    in_agenda = agenda.get(history)
                    if in_agenda is None or new_weight_sum > in_agenda.weight_sum:
                        agenda[history] = Beam(tags, new_weight_sum, features, beam)
            beams = sorted(agenda.values(), key=operator.attrgetter("weight_sum"), reverse=True)[:self.beam_size]
            if y is not None:
                gold_tags.append(y[i])
                if self.ignore_target is not None:
                    gold_not_in_beam = all(any(p != g and g != self.ignore_target_mapping for p, g in zip(beam.tags, gold_tags)) for beam in beams)
                else:
                    gold_not_in_beam = all(beam.tags != gold_tags for beam in beams)
                if gold_not_in_beam:
                    break
        return beams[0].tags, self._extract_feature_sequence(beams[0])

    def _predict_static(self, features):
        """"""
        weight_sum = np.zeros(self.target_size)
        weight_sum += np.sum([self.weights[feat] for feat in features if feat in self.weights], axis=0)
        if self.prior_weights is not None:
            weight_sum += np.sum([self.prior_weights[feat] for feat in features if feat in self.prior_weights], axis=0)
        return weight_sum

    def _predict_latent(self, features, static_weights):
        """"""
        weight_sum = np.sum([self.weights[feat] for feat in features if feat in self.weights], axis=0)
        if self.prior_weights is not None:
            weight_sum += np.sum([self.prior_weights[feat] for feat in features if feat in self.prior_weights], axis=0)
        weight_sum += static_weights
        predictions = np.argsort(weight_sum)[-self.beam_size:]
        return reversed(list(zip(predictions, weight_sum[predictions])))

    def _update(self, y, predicted, features, counter):
        """"""
        for feature_set, true_cls, predicted_cls in zip(features, y, predicted):
            if true_cls != predicted_cls:
                if self.ignore_target is not None and true_cls == self.ignore_target_mapping:
                    continue
                for feat in feature_set:
                    if feat not in self.weights:
                        self.weights[feat] = np.zeros(self.target_size)
                        self.weights_c[feat] = np.zeros(self.target_size)
                    self.weights[feat][true_cls] += 1
                    self.weights_c[feat][true_cls] += counter
                    self.weights[feat][predicted_cls] -= 1
                    self.weights_c[feat][predicted_cls] -= counter
            counter += 1

# def train_by_iterative_parameter_mixing(training_data, iterations=10, beam_size=5, n_shards=5):
#     """Iterative parameter mixing was first described by McDonald et al.
#     (2010).

#     """
#     weights = collections.defaultdict(lambda: collections.defaultdict(float))
#     weights_c = collections.defaultdict(lambda: collections.defaultdict(float))
#     counter = 0
#     random.seed(42)
#     random.shuffle(training_data)
#     div, mod = divmod(len(training_data), n_shards)
#     shards = [training_data[i * div + min(i, mod):(i + 1) * div + min(i + 1, mod)] for i in range(n_shards)]
#     for it in range(iterations):
#         ipm = functools.partial(ipm_iteration, it=it, beam_size=beam_size, w=weights, wc=weights_c, c=counter)
#         results = map(ipm, shards)
#         weights = collections.defaultdict(lambda: collections.defaultdict(float))
#         weights_c = collections.defaultdict(lambda: collections.defaultdict(float))
#         counter, correct, total, early_update = 0, 0, 0, 0
#         for w, wc, c, cor, tot, earl_upd in results:
#             for feat in w:
#                 for cls, weight in w[feat].items():
#                     weights[feat][cls] += weight / n_shards
#             for feat in wc:
#                 for cls, weight in wc[feat].items():
#                         weights_c[feat][cls] += weight / n_shards
#             counter += c / n_shards
#             correct += cor
#             total += tot
#             early_update += earl_upd
#         logging.info("Iteration %d: %d/%d = %.2f%% (%d early update)" % (it, correct, total, (correct / total) * 100, early_update))
#     # return weights - weights_c / counter
#     for feat in weights:
#         for cls in weights[feat]:
#             weights[feat][cls] -= weights_c[feat][cls] / counter
#     return weights


# def ipm_iteration(training_data, it, beam_size, w, wc, c):
#     """"""
#     weights = collections.defaultdict(lambda: collections.defaultdict(float))
#     weights_c = collections.defaultdict(lambda: collections.defaultdict(float))
#     counter = c
#     random.seed(it)
#     random.shuffle(training_data)
#     for feat in w:
#         for cls, weight in w[feat].items():
#             weights[feat][cls] = weight
#     for feat in wc:
#         for cls, weight in wc[feat].items():
#             weights_c[feat][cls] = weight
#     total, incorrect, early_update = 0, 0, 0
#     for structure in training_data:
#         items, static_features, gold_classes = structure
#         predicted, features = beam_search(items, static_features, weights, beam_size, gold_classes)
#         assert type(predicted) == type(gold_classes)
#         if len(predicted) != len(gold_classes):
#             early_update += 1
#         if predicted != gold_classes:
#             incorrect += 1
#             weights, weights_c = update(weights, weights_c, gold_classes, predicted, features, counter)
#         counter += len(predicted)
#         total += 1
#     correct = total - incorrect
#     return weights, weights_c, counter, correct, total, early_update
