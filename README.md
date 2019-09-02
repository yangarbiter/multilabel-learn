# multilabel-learn: Multilabel-Classification Algorithms

[![Build Status](https://travis-ci.org/yangarbiter/multilabel-learn.svg?branch=master)](https://travis-ci.org/yangarbiter/multilabel-learn)

## Implemented Algorithms

#### Cost-Sensitive Algorithms

* [RethinkNet](mlearn/models/rethinknet/rethinkNet.py): mlearn.models.RethinkNet
* [Cost-Sensitive Reference Pair Encoding (CSRPE)](mlearn/models/csrpe.py): mlearn.models.CSRPE
* [Probabilistic Classifier Chains](mlearn/models/probabilistic_classifier_chains.py): mlearn.models.ProbabilisticClassifierChains

#### Other Algorithms

* [Binary Relevance](mlearn/models/rethinknet/binary_relevance.py): mlearn.models.BinaryRelevance
* [Classifier Chains](mlearn/models/rethinknet/classifier_chains.py): mlearn.models.ClassifierChains
* [RAndom K labELsets](mlearn/models/rethinknet/random_k_labelsets.py): mlearn.models.RandomKLabelsets

## Installation

Compile and install the C-extensions
```bash
python ./setup.py install
```

Run example locally
```bash
pip install numpy Cython
python ./setup.py build_ext -i
PYTHONPATH=. python ./examples/classification.py
```
