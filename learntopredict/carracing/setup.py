#!/usr/bin/env python

from distutils.core import setup

setup(name='estool',
      version='1.0',
      description='Implementation of various Evolution Strategies',
      author='David Ha',
      author_email='hardmaru@gmail.com',
      url='https://github.com/hardmaru/estool',
      py_modules=['config', 'es', 'env', 'model', 'train'],
     )
