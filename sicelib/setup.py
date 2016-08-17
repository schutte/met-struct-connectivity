#!/usr/bin/env python

from numpy.distutils.core import setup, Extension

setup(name="sicelib",
        version="1.0",
        description="Sparse inverse covariance estimation for metabolic brain connectivity",
        author="Michael Schutte",
        author_email="michael.schutte@uiae.at",
        packages=["sicelib"],
        package_data={"sicelib": ["native/glasso.dll", "native/glasso.so"]},
        install_requires=["numpy", "scipy", "networkx"])
