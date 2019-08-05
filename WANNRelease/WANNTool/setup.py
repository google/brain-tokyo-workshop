#!/usr/bin/env python

from distutils.core import setup

setup(name='estool',
      version='1.0',
      description='Implementation of various Evolution Strategies',
      py_modules=['config', 'es', 'env', 'model', 'train'],
     )
