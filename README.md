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

## Citations

For RethinkNet, please cite
```bib
@article{yang2018deep,
  title={Deep learning with a rethinking structure for multi-label classification},
  author={Yang, Yao-Yuan and Lin, Yi-An and Chu, Hong-Min and Lin, Hsuan-Tien},
  journal={arXiv preprint arXiv:1802.01697},
  year={2018}
}
```

For Cost-Sensitive Reference Pair Encoding (CSRPE), please cite
```bib
@inproceedings{YY2018csrpe,
  title = {Cost-Sensitive Reference Pair Encoding for Multi-Label Learning},
  author = {Yao-Yuan Yang and Kuan-Hao Huang and Chih-Wei Chang and Hsuan-Tien Lin},
  booktitle = {Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year = 2018,
  arxiv = {https://arxiv.org/abs/1611.09461},
  software = {https://github.com/yangarbiter/multilabel-learn/blob/master/mlearn/models/csrpe.py},
}
```
