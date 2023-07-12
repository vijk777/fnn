#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="fnn",
    version="0.0.1",
    description="Foundation Neural Networks of the Neocortex",
    packages=find_packages(),
    install_requires=["torch"],
)
