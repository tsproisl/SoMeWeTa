#!/usr/bin/env python3

import collections
import itertools
import logging
import operator
import random

from nlp4py import utils

Beam = collections.namedtuple("Beam", ["tags", "weight_sum", "features"])


class AveragedStructuredPerceptron:
    """An averaged structured perceptron.

    The perceptron algorithm is due to Rosenblatt (1958). Freund and
    Schapire (1999: 292) found "that voting and averaging perform
    better than using the last vector". We implement averaging.
    Collins (2002) introduced the structured perceptron and Collins
    and Roark (2004) suggested the early update strategy.

    """
    def __init__(self, beam_size, iterations, latent_features, prior_weights=None):
        self.beam_size = beam_size
        self.iterations = iterations
        self.latent_features = latent_features
        self.prior_weights = prior_weights
        # self.weights = collections.defaultdict(lambda: collections.defaultdict(float))
        # self.weights_c = collections.defaultdict(lambda: collections.defaultdict(float))
        self.weights = {}
        self.weights_c = {}

    def fit(self, X, y, lengths):
        """"""
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
                if predicted != local_y:
                    incorrect += 1
                    self._update(local_y, predicted, features, counter)
                counter += len(predicted)
                total += 1
            random.seed(it)
            random.shuffle(ranges)
            correct = total - incorrect
            logging.info("Iteration %d: %d/%d = %.2f%% (%d early update)" % (it, correct, total, (correct / total) * 100, early_update))
        for feat in self.weights:
            for cls in self.weights[feat]:
                self.weights[feat][cls] -= self.weights_c[feat][cls] / counter
        if self.prior_weights is not None:
            for feat in self.prior_weights:
                for cls, weight in self.prior_weights[feat].items():
                    try:
                        self.weights[feat][cls] += weight
                    except KeyError:
                        if feat not in self.weights:
                            self.weights[feat] = {}
                        if cls not in self.weights[feat]:
                            self.weights[feat][cls] = weight

    def predict(self, X, lengths):
        """"""
        ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        for start, length in ranges:
            local_X = X[start:start + length]
            predicted, features = self._beam_search(local_X, start)
            yield predicted

    def score(self, X, y, lengths):
        """"""
        predicted = []
        ranges = list(zip((a - b for a, b in zip(itertools.accumulate(lengths), lengths)), lengths))
        for start, length in ranges:
            local_X = X[start:start + length]
            local_pred, features = self._beam_search(local_X, start)
            predicted.extend(local_pred)
        return utils.evaluate(y, predicted)

    def _beam_search(self, X, start, y=None):
        """"""
        beams = [Beam([], 0, [])]
        gold_tags = []
        for i, static_features in enumerate(X):
            agenda = []
            weight_sum = self._predict_static(static_features)
            for beam in beams:
                latent_features = self.latent_features(start, beam.tags, i)
                features = static_features | latent_features
                for prediction, weight in self._predict_latent(latent_features, weight_sum):
                    agenda.append(Beam(beam.tags + [prediction], beam.weight_sum + weight, beam.features + [features]))
            beams = sorted(agenda, key=operator.attrgetter("weight_sum"), reverse=True)[:self.beam_size]
            if y is not None:
                gold_tags.append(y[i])
                beam_tags = [beam.tags for beam in beams]
                if all(bt != gold_tags for bt in beam_tags):
                    break
        prediction, weights, features = beams[0]
        return prediction, features

    def _predict_static(self, features):
        """"""
        weight_sum = collections.defaultdict(float)
        weights = self.weights
        prior_weights = self.prior_weights
        for feat in features:
            if feat in weights:
                for cls, weight in weights[feat].items():
                    weight_sum[cls] += weight
            if prior_weights is not None and feat in prior_weights:
                for cls, weight in prior_weights[feat].items():
                    weight_sum[cls] += weight
        return weight_sum

    def _predict_latent(self, features, static_weights):
        """"""
        weight_sum = collections.defaultdict(float)
        for cls, weight in static_weights.items():
            weight_sum[cls] = weight
        weights = self.weights
        prior_weights = self.prior_weights
        for feat in features:
            if feat in weights:
                for cls, weight in weights[feat].items():
                    weight_sum[cls] += weight
            if prior_weights is not None and feat in prior_weights:
                for cls, weight in prior_weights[feat].items():
                    weight_sum[cls] += weight
        if len(weight_sum) > 0:
            # Add class labels to break ties
            return [_ for _ in sorted(weight_sum.items(), key=operator.itemgetter(1, 0), reverse=True)][:self.beam_size]
        else:
            return [(None, 0)]

    def _update(self, y, predicted, features, counter):
        """"""
        weights = self.weights
        weights_c = self.weights_c
        for feature_set, true_cls, predicted_cls in zip(features, y, predicted):
            if true_cls != predicted_cls:
                for feat in feature_set:
                    try:
                        weights[feat][true_cls] += 1
                    except KeyError:
                        if feat not in weights:
                            weights[feat] = {}
                            weights_c[feat] = {}
                        if true_cls not in weights[feat]:
                            weights[feat][true_cls] = 1
                            weights_c[feat][true_cls] = 0
                    weights_c[feat][true_cls] += counter
                    if predicted_cls is not None:
                        try:
                            weights[feat][predicted_cls] -= 1
                        except KeyError:
                            if predicted_cls not in weights[feat]:
                                weights[feat][predicted_cls] = -1
                                weights_c[feat][predicted_cls] = 0
                        weights_c[feat][predicted_cls] -= counter
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
