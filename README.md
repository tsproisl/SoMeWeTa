# SoMeWeTa #

  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Usage](#usage)
      * [Tagging a text](#tagging-a-text)
      * [Training the tagger](#training-the-tagger)
      * [Evaluating a model](#evaluating-a-model)
      * [Performing cross-validation](#performing-cross-validation)
  * [Model files](#model-files)
      * [German newspaper texts](#german_newspaper)
      * [German web and social media texts](#german_wsm)
      * [English newspaper texts](#english_newspaper)
  * [References](#references)


## Introduction ##

SoMeWeTa (short for Social Media and Web Tagger) is a part-of-speech
tagger that supports domain adaptation and that can incorporate
external sources of information such as Brown clusters and lexica. It
is based on the averaged structured perceptron and uses beam search
and an early update strategy. It is possible to train and evaluate the
tagger on partially annotated data.

SoMeWeTa achieves state-of-the-art results on the German web and
social media texts from the [EmpiriST 2015 shared
task](https://sites.google.com/site/empirist2015/) on automatic
linguistic annotation of computer-mediated communication / social
media. Therefore, SoMeWeTa is particularly well-suited to tag all
kinds of written German discourse, for example chats, forums, wiki
talk pages, tweets, blog comments, social networks, SMS and WhatsApp
dialogues.

The system is described in greater detail in [Proisl
(2018)](http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf).

For tokenization and sentence splitting on these kinds of text, we
recommend [SoMaJo](https://github.com/tsproisl/SoMaJo), a tokenizer
and sentence splitter with state-of-the-art performance on German web
and social media texts:

    somajo-tokenizer --split_sentences <file> | somewe-tagger --tag <model> -

In addition to the German web and social media model, we also provide
models trained on German and English newspaper texts. For both
languages, SoMeWeTa achieves highly competitive results close to the
current state of the art.


## Installation ##

SoMeWeTa can be easily installed using pip:

    pip3 install SoMeWeTa

Alternatively, you can download and decompress the
[latest release](https://github.com/tsproisl/SoMeWeTa/releases/latest)
or clone the git repository:

    git clone https://github.com/tsproisl/SoMeWeTa.git

In the new directory, run the following command:

    python3 setup.py install


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

If your machine has multiple cores, you can use the `--parallel`
option to speed up tagging. To tag a file using four cores, use this
command:

    somewe-tagger --parallel 4 --tag <model> <file>

Using the option `-x` or `--xml`, it is possible to tag an XML file.
The tagger assumes that each XML tag is on a separate line:

    somewe-tagger --xml --tag <model> <file>


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

It is also possible to train the tagger on partially annotated data.
To do this, assign a pseudo-tag to each unannotated token and tell
SoMeWeTa to ignore this pseudo-tag:

    somewe-tagger --train <model> --ignore-tag <pseudo-tag> <file>

Using the option `-x` or `--xml`, it is possible to train the tagger
on an XML file. It is assumed that each XML tag is on a separate line:

    somewe-tagger --xml --train <model> <file>


### Evaluating a model ###

To evaluate a model, you need an annotated input file in the same
format as for training. Then you can run the following command:

    somewe-tagger --evaluate <model> <file>

You can also evaluate a model on partially annotated data. Simply
assign a pseudo-tag to each unannotated token and tell SoMeWeTa to
ignore this pseudo-tag:

    somewe-tagger --evaluate <model> --ignore-tag <pseudo-tag> <file>

Using the option `-x` or `--xml`, it is possible to evaluate a model
on an XML file. The tagger assumes that each XML tag is on a separate
line:

    somewe-tagger --xml --evaluate <model> <file>


### Performing cross-validation ###

You can also perform a 10-fold cross-validation on a training corpus:

    somewe-tagger --crossvalidate <file>

To perform a cross-validation on partially annotated data, assign a
pseudo-tag to each unannotated token and tell SoMeWeTa to ignore this
pseudo-tag:

    somewe-tagger --crossvalidate --ignore-tag <pseudo-tag> <file>

Using the option `-x` or `--xml`, it is possible to perform a
cross-validation on an XML file. The tagger assumes that each XML tag
is on a separate line:

    somewe-tagger --xml --crossvalidate <file>


## Model files ##

| Model                                      | tagset       | est. accuracy |
|--------------------------------------------|--------------|---------------|
| [German newspaper](#german_newspaper)      | STTS (TIGER) |        97.98% |
| [German web and social media](#german_wsm) | STTS\_IBK    |        91.42% |
| [English newspaper](#english_newspaper)    | Penn         |        97.25% |


### German newspaper texts <a id="german_newspaper"/> ###

This model has been trained on the entire [TIGER
corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.html)
and uses Brown clusters extracted from
[DECOW14](http://corporafromtheweb.org/decow14/) and coarse
wordclasses [extracted](http://www.danielnaber.de/morphologie/) from
[Morphy](http://morphy.wolfganglezius.de/) as additional information.

To estimate the accuracy of this model, we performed a 10-fold
cross-validation on the TIGER corpus with the same settings, resulting
in a mean accuracy plus or minus two standard deviations of 97.98%
±0.32.

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2017-09-14.model)
(115 MB) – Note that according to the [TIGER Corpus License
agreement](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/htmlicense.html)
“use of data derived from the corpus for any commercial purposes
requires explicit written agreement of Licenser.”


### German web and social media texts <a id="german_wsm"> ###

This model uses a [variant of the above
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_for_empirist_2017-12-20.model)
as prior and is trained on the entire [data from the EmpiriST 2015
shared task](https://sites.google.com/site/empirist2015/home/gold),
i.e. both the training and the test data, as well as a little bit of
additional training data (cf. the data directory of this repository).
It uses the same additional sources of information as the prior model.

A variant of this model that only uses the training part of the
EmpiriST 2015 data achieves a mean accuracy of 91.42% on the two test
sets:

| Corpus | all words   | known words | unknown words |
|--------|-------------|-------------|---------------|
| CMC    | 89.08 ±0.25 | 90.95 ±0.27 | 77.41 ±1.14   |
| Web    | 93.77 ±0.26 | 95.34 ±0.25 | 83.31 ±0.63   |

As of December 2017, those figures represent the state of the art on
the EmpiriST data.

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_web_social_media_2017-12-20.model)
(115 MB) – Note that according to the [TIGER Corpus License
agreement](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/htmlicense.html)
“use of data derived from the corpus for any commercial purposes
requires explicit written agreement of Licenser.”


### English newspaper texts <a id="english_newspaper"> ###

This model has been trained on all sections of the Wall Street Journal
part of the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42)
and uses Brown clusters extracted from
[ENCOW14](http://corporafromtheweb.org/encow14/) and part-of-speech
data extracted from the [English DELA
dictionary](http://infolingu.univ-mlv.fr/DonneesLinguistiques/Dictionnaires/telechargement.html)
as additional information.

A variant of this model that was trained only on sections 0–18 of the
Wall Street Journal achieves the following results on the usual
development and test sets:

| Data set     | all words   | known words | unknown words |
|--------------|-------------|-------------|---------------|
| dev (19–21)  | 97.15 ±0.02 | 97.41 ±0.03 | 89.59 ±0.28   |
| test (22–24) | 97.25 ±0.02 | 97.42 ±0.03 | 91.05 ±0.29   |

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/english_newspaper_2017-09-15.model)
(38 MB) – You can use this model for research purposes if you have a
license for the Penn Treebank. If you want to use it for commercial
applications, please contact the Linguistic Data Consortium to find
out which conditions apply.


## References ##

  * Proisl, Thomas (2018): “SoMeWeTa: A Part-of-Speech Tagger for
    German Social Media and Web Texts.” In: Proceedings of the
    Eleventh International Conference on Language Resources and
    Evaluation (LREC 2018). Miyazaki: European Language Resources
    Association (ELRA), 665–670.
    [PDF](http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf).
