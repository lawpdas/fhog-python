#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('fhog', sources = ['fhog.cpp']) ]

setup(
        name = 'python MATLAB fhog',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )
