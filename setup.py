#!/usr/bin/env python

import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


install_requirements = [
    'numpy>=1.9.0',
    'scipy>=0.19.0'
]

setup(
    name='texturesynth',
    version='0.0.1',
    description="Tools for texture analysis and synthesis"
    "with Simoncelli's method",
    author='Aiga SUZUKI',
    author_email='ai-suzuki@aist.go.jp',
    license='MIT License',
    packages=['texturesynth'],
    package_dir={'texturesynth': 'texturesynth'},
    setup_requires=[],
    install_requires=install_requirements,
)
