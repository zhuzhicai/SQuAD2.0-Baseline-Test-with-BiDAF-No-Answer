# BiDAF-No-Answer
This repo was copied from [https://bitbucket.org/omerlevy/bidaf_no_answer](https://bitbucket.org/omerlevy/bidaf_no_answer).

## Setup
To get the BNA server running, you need a `glove` directory in the root directory, 
containing the standard GloVe files.

# Original README

# Extending the Bi-directional Attention Flow Model with "No Answer"

- This is the implementation used in [Zero-Shot Relation Extraction via Reading Comprehension][paper1] (Levy et al., 2017).
- It is an extension of the [BiDAF model][paper2] by Seo et al.
- This file describes some basic use-cases in the relation-extraction setting.  The original implementation's readme file is [BiDAF_README.md][fullreadme].

## Requirements
- Python (developed on 3.5.2. Issues have been reported with Python 2!)
- tensorflow (deep learning library, verified on r0.11)
- nltk (NLP tools, verified on 3.2.1)
- tqdm (progress bar, verified on 4.7.4)

## Scripts
- `run_prep.sh <run name>` calls an internal script (`zeroshot2squad.py`) that changes our tab-delimited format to SQuAD's JSON format. It then performs any necessary preprocessing for the BiDAF model.
- `run_train.sh <run name>` runs the training procedure.
- `run_test.sh <run name>` runs the testing procedure, and yields an answer file in `out/basic/<run name>/test-#####.json`
- `python analyze.py <test set> <answer file>` reads the test set and the model's answers, and returns the F1 score broken down by different factors.


[paper1]: 
[paper2]: https://arxiv.org/abs/1611.01603
[fullreadme]: BiDAF_README.md

