# SoMeWeTa #

[![PyPI](https://img.shields.io/pypi/v/SoMeWeTa)](https://pypi.org/project/SoMeWeTa/)

  * [Introduction](#introduction)
  * [Installation](#installation)
  * [Usage](#usage)
      * [Tagging a text](#tagging-a-text)
      * [Training the tagger](#training-the-tagger)
      * [Evaluating a model](#evaluating-a-model)
      * [Performing cross-validation](#performing-cross-validation)
      * [Using the module](#using-the-module)
  * [Model files](#model-files)
      * [German newspaper texts](#german_newspaper)
      * [German web and social media texts](#german_wsm)
      * [English newspaper texts](#english_newspaper)
      * [French newspaper texts](#french_newspaper)
      * [Spoken Italian](#spoken_italian)
      * [Bhojpuri](#bhojpuri)
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
models trained on German, English and French newspaper texts, as well
as models for Bhojpuri and spoken Italian. For all languages, SoMeWeTa
achieves highly competitive results close to the current state of the
art.


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

When called with the `--progress` option, SoMeWeTa displays tagging
progress, average and current tagging speed and remaining time.

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


### Using the module ###

To incorporate the tagger into your own Python project, you have to
import `someweta.ASPTagger`, create an `ASPTagger` object, load a
pretrained model and call the `tag_sentence` method:

```python
from someweta import ASPTagger

model = "german_web_social_media_2018-12-21.model"
sentences = [["Ein", "Satz", "ist", "eine", "Liste", "von", "Tokens", "."],
             ["Zeitfliegen", "mögen", "einen", "Pfeil", "."]]

asptagger = ASPTagger()
asptagger.load(model)

for sentence in sentences:
    tagged_sentence = asptagger.tag_sentence(sentence)
    print("\n".join(["\t".join(t) for t in tagged_sentence]), "\n", sep="")
```

Here is an example for using SoMaJo and SoMeWeTa in combination,
performing tokenization, sentence splitting and part-of-speech
tagging:

```python
import somajo
import someweta

filename = "test.txt"
model = "german_web_social_media_2018-12-21.model"

asptagger = someweta.ASPTagger()
asptagger.load(model)

# See https://github.com/tsproisl/SoMaJo#using-the-module
tokenizer = somajo.SoMaJo("de_CMC", split_camel_case=False)
sentences = tokenizer.tokenize_text_file(filename, paragraph_separator="empty_lines")
for sentence in sentences:
    tokens = [token.text for token in sentence]
    tagged_sentence = asptagger.tag_sentence(tokens)
    print("\n".join("\t".join(t) for t in tagged_sentence), "\n", sep="")
```

## Model files ##

| Model                                      | tagset       | est. accuracy |
|--------------------------------------------|--------------|---------------|
| [German newspaper](#german_newspaper)      | STTS (TIGER) | 98.02%        |
| [German web and social media](#german_wsm) | STTS\_IBK    | 92.18%        |
| [English newspaper](#english_newspaper)    | Penn         | 97.25%        |
| [French newspaper](#french_newspaper)      | FTB-29       | 97.71%        |
| [Spoken Italian](#spoken_italian)          | UD (KIPoS)   | 91.79%        |
| [Bhojpuri](#bhojpuri)                      | BIS-33       | 92.58%        |


### German newspaper texts <a id="german_newspaper"/> ###

This model has been trained on the entire [TIGER
corpus](http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/tiger.html)
and uses Brown clusters (extracted from
[DECOW16AX](http://corporafromtheweb.org/decow16/),
[GeRedE](https://github.com/fau-klue/german-reddit-korpus) and a
collection of German tweets) and coarse wordclasses
[extracted](http://www.danielnaber.de/morphologie/) from
[Morphy](http://morphy.wolfganglezius.de/) as additional information.

To estimate the accuracy of this model, we performed a 10-fold
cross-validation on the TIGER corpus with the same settings, resulting
in a 95% confidence interval of 98.02% ±0.12.

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model)
(111 MB) – Note that the model is provided for research purposes only.
For further information, please refer to the licenses of the
individual resources that were used in the creation of the model.


### German web and social media texts <a id="german_wsm"> ###

This model uses a [variant of the above
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_for_empirist_2020-05-28.model)
as prior and is trained on the entire [EmpiriST 2.0
corpus](https://github.com/fau-klue/empirist-corpus), i.e. both the
training and the test data, as well as a little bit of additional
training data (cf. the data directory of this repository). It uses the
same additional sources of information as the prior model.

A variant of this model that only uses the training part of the
EmpiriST corpus achieves a mean accuracy of 92.18% on the two test
sets:

| Corpus | all words   | known words | unknown words |
|--------|-------------|-------------|---------------|
| CMC    | 90.39 ±0.30 | 92.42 ±0.29 | 77.57 ±1.40   |
| Web    | 93.96 ±0.16 | 95.56 ±0.17 | 83.40 ±0.69   |

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/german_web_social_media_2020-05-28.model)
(112 MB) – Note that the model is provided for research purposes only.
For further information, please refer to the licenses of the
individual resources that were used in the creation of the model.


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
(38 MB) – Note that the model is provided for research purposes only.
For further information, please refer to the licenses of the
individual resources that were used in the creation of the model.


### French newspaper texts <a id="french_newspaper"> ###

This model has been trained on the [French
Treebank](http://ftb.linguist.univ-paris-diderot.fr/) and uses Brown
clusters extracted from
[FRCOW16](http://corporafromtheweb.org/frcow16/) and part-of-speech
data extracted from the [French DELA
dictionary](http://infolingu.univ-mlv.fr/DonneesLinguistiques/Dictionnaires/telechargement.html)
as additional information.

The French Treebank is annotated with two different tagsets: A
coarse-grained tagset consisting of 15 tags and a more fine-grained
tagset consisting of 29 tags. The model has been trained on the more
fine-grained tagset. However, we provide a mapping to the smaller
tagset (`data/mapping_french_29_to_15.json`) that can be used to
annotate a text with both tagsets:

    somewe-tagger --tag <model> --mapping <mapping> <file>

To estimate the accuracy of the model, we performed a 10-fold
cross-validation on the French Treebank using the same settings:

| tagset           | accuracy    |
|------------------|-------------|
| 29 tags          | 97.71 ±0.13 |
| 15 tags (mapped) | 98.22 ±0.11 |

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/french_newspaper_2018-06-20.model)
(28 MB) – Note that the model is provided for research purposes only.
For further information, please refer to the licenses of the
individual resources that were used in the creation of the model.


### Spoken Italian <a id="spoken_italian"> ###

<!-- ten-fold crossvalidation on the whole dataset: 92.50% ±0.92 -->

This model has been pretrained on the union of all Italian corpora in
the [Universal Dependencies
project](https://universaldependencies.org/) and then been adapted to
spoken Italian using [annotated data from the KIParla
corpus](https://github.com/boscoc/kipos2020). The model uses
coarse-grained wordclass information from
[Morph-it!](https://docs.sslmit.unibo.it/doku.php?id=resources:morph-it)
and Brown clusters extracted from a collection of Italian corpora
([OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles-v2018.php), Reddit
posts, [PAISÀ](http://www.corpusitaliano.it/), Wikimedia dumps,
[OSCAR](https://oscar-corpus.com/)). The input text must be tokenized
according to the [UD tokenization
guidelines](https://universaldependencies.org/u/overview/tokenization.html).
In particular, the model expects that contracted forms like *parlarmi*
(*parlar* + *mi*) or *della* (*di* + *la*) are split into their
constituents. A detailed description and analysis of the model A
detailed description and analysis of the model is available in [Proisl
and Lapesa (2020)](http://ceur-ws.org/Vol-2765/paper140.pdf).

A variant of this model that only uses the training part of the
KIParla corpus achieves a mean accuracy of 91.79% on the two test
sets:

| Corpus   | all words | known words | unknown words |
|----------|-----------|-------------|---------------|
| formal   | 92.67     | 93.39       | 67.92         |
| informal | 90.90     | 91.41       | 75.00         |

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/spoken_italian_2021-02-26.model)
(43 MB) – Note that the model is provided for research purposes only.
For further information, please refer to the licenses of the
individual resources that were used in the creation of the model.


### Bhojpuri <a id="bhojpuri"> ###

<!-- ten-fold crossvalidation on the whole dataset: 92.32% ±0.72 -->

This model has been trained on ca. 105,000 tokens of annotated
Bhojpuri text provided by the organizers of the [NSURL shared task for
Bhojpuri](http://nsurl.org/2019-2/tasks/task-10-low-level-nlp-tools-for-bhojpuri-language/).
Additionally, the model uses Brown clusters extracted from text
collections of related languages (Hindi and Bihari Wikimedia dumps and
a [Magahi corpus](https://github.com/kmi-linguistics/magahi)). The
model uses a fine-grained variant of the Bureau of Indian Standards
(BIS) annotation scheme with 33 tags. A more detailed description of
the model can be found in [Proisl et al.
(2019)](https://www.aclweb.org/anthology/2019.nsurl-1.11).

A variant of this model that only uses the training part of the
dataset achieves an accuracy of 92.58% on the test set:

| All words | known words | unknown words |
|-----------|-------------|---------------|
| 92.58     | 94.57       | 75.09         |

[Download
model](http://corpora.linguistik.uni-erlangen.de/someweta/bhojpuri_2021-02-26.model)
(3,7 MB) – Note that the model is provided for research purposes only.
For further information, please refer to the licenses of the
individual resources that were used in the creation of the model.


## References ##

  * If you use **SoMeWeTa** for academic research, please consider
    citing the following paper:
  
    Proisl, Thomas. 2018. “SoMeWeTa: A Part-of-Speech Tagger for
    German Social Media and Web Texts.” In *Proceedings of the
    Eleventh International Conference on Language Resources and
    Evaluation (LREC 2018)*, 665–670. Miyazaki: European Language
    Resources Association (ELRA).
    [PDF](http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf).

	```bibtex
    @InProceedings{Proisl_LREC:2018,
      author    = {Proisl, Thomas},
      title     = {{SoMeWeTa}: {A} Part-of-Speech Tagger for {G}erman Social Media and Web Texts},
      booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)},
      year      = {2018},
      address   = {Miyazaki},
      publisher = {European Language Resources Association {ELRA}},
      pages     = {665--670},
      url       = {http://www.lrec-conf.org/proceedings/lrec2018/pdf/49.pdf},
    }
	```
  * If you use the model for **spoken Italien**, please consider also
    citing the following paper:
  
    Proisl, Thomas, and Gabriella Lapesa. 2020. “KLUMSy@KIPoS:
    Experiments on Part-of-Speech Tagging of Spoken Italian.” In
    *Proceedings of the 7th Evaluation Campaign of Natural Language
    Processing and Speech Tools for Italian (EVALITA 2020)*.
    CEUR-WS.org. [PDF](http://ceur-ws.org/Vol-2765/paper140.pdf).

    ```bibtex
    @InProceedings{Proisl_Lapesa_EVALITA:2020,
      author    = {Proisl, Thomas and Lapesa, Gabriella},
      title     = {{KLUMSy@KIPoS}: Experiments on Part-of-Speech Tagging of Spoken {I}talian},
      booktitle = {Proceedings of the 7th Evaluation Campaign of Natural Language Processing and Speech Tools for {I}talian ({EVALITA} 2020)},
      year      = {2020},
      editor    = {Basile, Valerio and Croce, Danilo and Di Maro, Maria and Passaro, Lucia C.},
      address   = {Online},
      publisher = {CEUR-WS.org},
      url       = {http://ceur-ws.org/Vol-2765/paper140.pdf}
    }
    ```
  * If you use the **Bhojpuri** model, please consider also citing the
    following paper:
  
    Proisl, Thomas, Peter Uhrig, Philipp Heinrich, Andreas Blombach,
    Sefora Mammarella, Natalie Dykes, and Besim Kabashi. 2019.
    “The_Illiterati: Part-of-Speech Tagging for Magahi and Bhojpuri
    Without Even Knowing the Alphabet.” In *Proceedings of the First
    International Workshop on NLP Solutions for Under Resourced
    Languages (NSURL 2019)*, 73–79. Trento: Association for
    Computational Linguistics.
    [PDF](https://www.aclweb.org/anthology/2019.nsurl-1.11).

    ```bibtex
    @InProceedings{Proisl_et_al_NSURL:2019,
      author    = {Proisl, Thomas and Uhrig, Peter and Heinrich, Philipp and Blombach, Andreas and Mammarella, Sefora and Dykes, Natalie and Kabashi, Besim},
      title     = {{T}he\_{I}lliterati: Part-of-Speech Tagging for {M}agahi and {B}hojpuri without Even Knowing the Alphabet},
      booktitle = {Proceedings of the First International Workshop on {NLP} Solutions for Under Resourced Languages ({NSURL} 2019)},
      year      = {2019},
      pages     = {73--79},
      address   = {Trento},
      publisher = {Association for Computational Linguistics},
      url       = {https://www.aclweb.org/anthology/2019.nsurl-1.11}
    }
    ```
