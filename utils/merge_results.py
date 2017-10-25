#!/usr/bin/env python3

import re
import statistics

accuracies = re.compile(r"^Accuracy: (?P<acc>\d\d.\d\d)%; IV: (?P<iv>\d\d.\d\d)%; OOV: (?P<oov>\d\d.\d\d)%$")

files = [open("results.%d.txt" % i) for i in range(10)]
contents = [f.readlines() for f in files]
aligned_lines = zip(*contents)
for lines in aligned_lines:
    acc, iv, oov = [], [], []
    if lines[0].startswith("Accuracy"):
        for line in lines:
            match = accuracies.search(line)
            acc.append(float(match.group("acc")))
            iv.append(float(match.group("iv")))
            oov.append(float(match.group("oov")))
        avg_acc = statistics.mean(acc)
        stdev_acc = statistics.stdev(acc)
        avg_iv = statistics.mean(iv)
        stdev_iv = statistics.stdev(iv)
        avg_oov = statistics.mean(oov)
        stdev_oov = statistics.stdev(oov)
        # print("| %.2f ±%.2f | %.2f ±%.2f | %.2f ±%.2f |" % (avg_acc, 2 * stdev_acc, avg_iv, 2 * stdev_iv, avg_oov, 2 * stdev_oov))
        print("& %.2f ±%.2f & %.2f ±%.2f & %.2f ±%.2f" % (avg_acc, 2 * stdev_acc, avg_iv, 2 * stdev_iv, avg_oov, 2 * stdev_oov))
    else:
        print(lines[0].rstrip())

for f in files:
    f.close()
