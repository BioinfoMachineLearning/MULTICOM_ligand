#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="FlowDock",
    version="0.0.1",
    description="Geometric flow matching for switch-state protein-ligand docking",
    author="",
    author_email="",
    url="https://github.com/BioinfoMachineLearning/FlowDock",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
