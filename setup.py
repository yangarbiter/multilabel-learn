#!/usr/bin/env python
from distutils.core import setup
import numpy
from Cython.Build import cythonize

setup(
  name = 'calc_score',
  ext_modules = cythonize("./mlearn/criteria/calc_score.pyx"),
  include_dirs=[numpy.get_include()]
)
