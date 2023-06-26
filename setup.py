#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="fnn",
    version="0",
    description="Foundation Neural Networks of the Neocortex",
    author="Eric Y. Wang",
    author_email="eric.wang2@bcm.edu",
    packages=find_packages(),
    install_requires=["torch"],
)
