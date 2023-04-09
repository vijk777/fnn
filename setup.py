#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='foundation-networks',
    version='0',
    description='Foundation networks of the visual cortex',
    author='Eric Y. Wang',
    author_email='eric.wang2@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=[],
)
