#!/usr/bin/env python
import os
from setuptools import setup, Extension
import sys

import numpy
from Cython.Build import cythonize

include_dirs = [numpy.get_include()]

extensions = cythonize([
    Extension(
        "mlearn.criteria.sparse_criteria",
        sources=["mlearn/criteria/sparse_criteria.pyx"],
        include_dirs=include_dirs,
    ),
    Extension(
        "mlearn.criteria.reweight",
        sources=["mlearn/criteria/reweight.pyx"],
        include_dirs=include_dirs,
    ),
    Extension(
        "mlearn.criteria.sparse_reweight",
        sources=["mlearn/criteria/sparse_reweight.pyx"],
        include_dirs=include_dirs,
    ),
])

with open('./requirements.txt') as f:
    requirements = f.read().splitlines()
install_requires = requirements

setup(
    name = 'multilabel-learn',
    version='0.0.1a0',
    description="",
    author='Y.-Y. Yang',
    url='https://github.com/yangarbiter/multilabel-learn',
    install_requires=install_requires,
    test_suite='mlearn',
    packages=[
        'mlearn',
        'mlearn.criteria',
    ],
    package_dir={
        'mlearn': 'mlearn',
        'mlearn.criteria': 'mlearn/criteria/',
    },
    ext_modules = extensions,
)
