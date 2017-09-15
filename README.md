# SoMeWeTa #

## Introduction ##

SoMeWeTa is a part-of-speech tagger that supports domain adaptation
and that can incorporate external sources of information such as Brown
clusters and lexica. It is based on the averaged structured perceptron
and uses beam search and an early update strategy.

SoMeWeTa achieves state-of-the-art results on the German web and
social media texts from the [EmpiriST 2015 shared
task](https://sites.google.com/site/empirist2015/) on automatic
linguistic annotation of computer-mediated communication / social
media. Therefore, SoMeWeTa is particularly well-suited to tag all
kinds of written German discourse, for example chats, forums, wiki
talk pages, tweets, blog comments, social networks, SMS and WhatsApp
dialogues.

In addition, we also provide models trained on German and English
newspaper texts. For both languages, SoMeWeTa achieves highly
competitive results close to the current state of the art.


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

| Model                                      | tagset    | est. accuracy |
|--------------------------------------------|-----------|---------------|
| [German newspaper](#german_newspaper)      | STTS      |        97.98% |
| [German web and social media](#german_wsm) | STTS\_IBK |        91.20% |
| [English newspaper](#english_newspaper)    | Penn      |        97.28% |


### German newspaper texts <a id="german_newspaper"/> ###

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

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper.model)
(115 MB)


### German web and social media texts <a id="german_wsm"> ###

This model uses the above model as prior and is trained on the entire
[data from the EmpiriST 2015 shared
task](https://sites.google.com/site/empirist2015/home/gold), i.e. both
the training and the test data. It uses the same additional sources of
information as the prior model.

Note that according to the [TIGER Corpus License
agreement](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/htmlicense.html)
“use of data derived from the corpus for any commercial purposes
requires explicit written agreement of Licenser.”

A variant of this model that was trained only on the EmpiriST 2015
training data achieves a mean accuracy of 91.20% on the two test sets:

| Corpus | all words   | known words | unknown words |
|--------|-------------|-------------|---------------|
| CMC    | 88.69 ±0.40 | 90.62 ±0.35 | 76.74 ±1.51   |
| Web    | 93.71 ±0.19 | 95.28 ±0.23 | 83.36 ±0.81   |

As of September 2017, those figures represent the state of the art on
the EmpiriST data.

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_web_social_media.model)
(116 MB)


### English newspaper texts <a id="english_newspaper"> ###

This model has been trained on all sections of the Wall Street Journal
part of the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42)
and uses Brown clusters extracted from
[ENCOW14](http://corporafromtheweb.org/encow14/) and part-of-speech
data extracted from the [English DELA
dictionary](http://infolingu.univ-mlv.fr/DonneesLinguistiques/Dictionnaires/telechargement.html)
as additional information.

You can use this model for research purposes if you have a license for
the Penn Treebank. If you want to use it for commercial applications,
please contact the Linguistic Data Consortium to find out which
conditions apply.

A variant of this model that was trained only on sections 0–18 of the
Wall Street Journal achieves the following results on the usual
development and test sets:

| Data set     | all words | known words | unknown words |
|--------------|-----------|-------------|---------------|
| dev (19–21)  | 97.16     | 97.41       | 89.89         |
| test (22–24) | 97.28     | 97.45       | 91.14         |

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/english_newspaper.model)
(38 MB)
