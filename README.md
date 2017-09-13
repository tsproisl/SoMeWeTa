# SoMeWeTa #

## Introduction ##

SoMeWeTa is a state-of-the-art part-of-speech tagger for German web
and social media texts that follows the guidelines of the [EmpiriST
2015 shared task](https://sites.google.com/site/empirist2015/) on
automatic linguistic annotation of computer-mediated communication /
social media. SoMeWeTa is particularly well-suited to tag all kinds of
written discourse, for example chats, forums, wiki talk pages, tweets,
blog comments, social networks, SMS and WhatsApp dialogues.


## Installation ##

SoMeWeTa can be easily installed using pip:

    pip install SoMeWeTa


## Usage ##

You can use the tagger as a standalone program from the command line.
General usage information is available via the `-h` option:

    somewe-tagger -h


### Tagging a text ###

SoMeWeTa requires that the input texts are tokenized and split into
sentences. Tokenization and sentence splitting have to be consistent
with the corpora the tagger model has been trained on. For German
texts, we recommend [SoMaJo](https://github.com/tsproisl/SoMaJo), a
tokenizer and sentence splitter with state-of-the-art performance on
German web and social media texts. The expected input format is one
token per line with an empty line after each sentence.

To tag a file, run the following command:

    somewe-tagger --tag <model> <file>


### Training the tagger ###

The expected input format for training the tagger is one token-pos
pair per line, where token and pos are seperated by a tab character,
and an empty line after each sentence. To train a model, run the
following command:

    somewe-tagger --train <model> <file>

SoMeWeTa supports domain adaptation. First train a model on the
background corpus, then use this model as prior when training the
in-domain model:

    somewe-tagger --train <model> --prior <background_model> <file>
	
SoMeWeTa can make use of additional sources of information. You can
use the `--brown` option to provide a file with Brown clusters (the
`paths` file produced by
[wcluster](https://github.com/percyliang/brown-cluster)) and the
`--lexicon` option to provide a lexicon with additional token-level
information. The lexicon should consist of lines with tab-separated
token-value pairs, e.g.:

    welcome	ADJ
	welcome	INTJ
	welcome	NOUN
	welcome	VERB
	work	NOUN
	work	VERB


### Evaluating a model ###

To evaluate a model, you need an annotated input file in the same
format as for training. Then you can run the following command:

    somewe-tagger --evaluate <model> <file>


### Performing cross-validation ###

You can also perform a 10-fold cross-validation on a training corpus:

    somewe-tagger --crossvalidate <file>


## Model files ##

### German newspaper texts ###

This model has been trained on the entire [TIGER
corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.html)
and uses Brown clusters extracted from
[DECOW14](http://corporafromtheweb.org/decow14/) and coarse
wordclasses [extracted](http://www.danielnaber.de/morphologie/) from
[Morphy](http://morphy.wolfganglezius.de/) as additional information.

Note that according to the [TIGER Corpus License
agreement](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/htmlicense.html)
“use of data derived from the corpus for any commercial purposes
requires explicit written agreement of Licenser.”

To estimate the accuracy of this model, we performed a 10-fold
cross-validation on the TIGER corpus with the same settings, resulting
in a mean accuracy plus or minus two standard deviations of 97.98%
±0.32.

TODO: download link


### German web and social media texts ###

This model uses the above model as prior and is trained on the entire
[data from the EmpiriST 2015 shared
task](https://sites.google.com/site/empirist2015/home/gold), i.e. both
the training and the test data. It uses the same additional sources of
information as the prior model.

Note that according to the [TIGER Corpus License
agreement](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/htmlicense.html)
“use of data derived from the corpus for any commercial purposes
requires explicit written agreement of Licenser.”

TODO: cross-validation results 88.19% ±1.72, 88.24% ±1.93, 88.26% ±2.29

TODO: download link

A variant of this model that was not trained on the EmpiriST 2015 test
data achieves a mean accuracy of 91.20% on those test sets:

| Corpus | all words   | known words | unknown words |
|--------|-------------|-------------|---------------|
| CMC    | 88.69 ±0.40 | 90.62 ±0.35 | 76.74 ±1.51   |
| Web    | 93.71 ±0.19 | 95.28 ±0.23 | 83.36 ±0.81   |

